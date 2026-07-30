from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridWorkerConfig:
    role: str
    algorithm: SpeculativeAlgorithm
    overrides: dict[str, Any]


@dataclass(frozen=True)
class HybridRuntimeState:
    """Shape-bound target resources for one Hybrid route.

    This follows adaptive decoding's runtime-state model: the backend that
    prepares verify metadata and the CUDA graph that replays it are activated
    together at a batch boundary.  The target KV/memory pools remain shared.
    """

    route: str
    speculative_num_draft_tokens: int
    target_attn_backend: Any
    target_graph_runner: Any


@dataclass
class HybridRouteStats:
    """Per-route scheduling and acceptance metrics, reset on flush_cache."""

    # Per-route batch-level counters. Each forward contributes one batch mean,
    # so a dynamic batch size does not dominate the routing comparison.
    retrieval_forward_ct: int = 0
    retrieval_num_accept_tokens: float = 0.0
    retrieval_num_draft_tokens: int = 0
    neural_forward_ct: int = 0
    neural_num_accept_tokens: float = 0.0
    neural_num_draft_tokens: int = 0

    # Retrieval routing-decision helpers, accumulated over pure decode batches.
    total_continuation_length: float = 0.0
    matching_req_ct: int = 0

    def reset(self) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, 0)

    @property
    def retrieval_accept_length(self) -> float:
        return (
            self.retrieval_num_accept_tokens / self.retrieval_forward_ct
            if self.retrieval_forward_ct
            else 0.0
        )

    @property
    def retrieval_accept_rate(self) -> float:
        return (
            (self.retrieval_num_accept_tokens - self.retrieval_forward_ct)
            / self.retrieval_num_draft_tokens
            if self.retrieval_num_draft_tokens
            else 0.0
        )

    @property
    def neural_accept_length(self) -> float:
        return (
            self.neural_num_accept_tokens / self.neural_forward_ct
            if self.neural_forward_ct
            else 0.0
        )

    @property
    def neural_accept_rate(self) -> float:
        return (
            (self.neural_num_accept_tokens - self.neural_forward_ct)
            / self.neural_num_draft_tokens
            if self.neural_num_draft_tokens
            else 0.0
        )

    @property
    def avg_continuation_length(self) -> float:
        return (
            self.total_continuation_length / self.total_forward_ct
            if self.total_forward_ct
            else 0.0
        )

    @property
    def total_forward_ct(self) -> int:
        return self.retrieval_forward_ct + self.neural_forward_ct


class HybridController(BaseSpecWorker):
    """JSON-configured, batch-level speculative worker router.

    Each configured worker owns an independent ServerArgs copy, so draft-token
    counts and other algorithm-specific settings are allowed to differ. The
    controller only assumes that the configured retrieval worker is NGRAM and
    that the neural worker exposes a draft-extend phase; worker classes
    themselves are always resolved through ``SpeculativeAlgorithm.create_worker``.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self._target_worker = target_worker
        config = self._load_config(server_args.speculative_hybrid_config)
        self.routing = {
            "min_continuation_ratio": config["min_continuation_ratio"],
            "min_matching_ratio": config["min_matching_ratio"],
        }
        self.worker_configs = self._parse_worker_configs(config)
        self.workers = self._create_workers(
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
        )
        self.retrieval_worker = self.workers["retrieval"]
        self.neural_worker = self.workers["neural"]
        if getattr(
            type(self.retrieval_worker), "get_retrieval_continuation_lengths"
        ) is getattr(BaseSpecWorker, "get_retrieval_continuation_lengths"):
            raise ValueError(
                "HYBRID retrieval worker must override get_retrieval_continuation_lengths."
            )
        if getattr(
            type(self.neural_worker), "update_runtime_state_from_hybrid"
        ) is getattr(BaseSpecWorker, "update_runtime_state_from_hybrid"):
            raise ValueError(
                "HYBRID neural worker must override update_runtime_state_from_hybrid."
            )
        if getattr(type(self.retrieval_worker), "update_runtime_state_from_hybrid") is getattr(
            BaseSpecWorker, "update_runtime_state_from_hybrid"
        ):
            raise ValueError("HYBRID retrieval worker must override update_runtime_state_from_hybrid.")
        self.speculative_num_steps = self.neural_worker.speculative_num_steps
        self._accept_width = max(
            self.retrieval_worker.speculative_num_draft_tokens,
            self.neural_worker.speculative_num_draft_tokens,
        )
        # Scheduler-facing allocation uses the largest route width. Each
        # producer keeps its own accepted-path row width on VerifyInput.
        self.speculative_num_draft_tokens = self._accept_width
        self._hybrid_draft_extend_widths = tuple(
            sorted(
                {
                    self.retrieval_worker.speculative_num_draft_tokens,
                    self.neural_worker.speculative_num_draft_tokens,
                }
            )
        )
        self._last_route = "neural"
        # Scheduler queries this before preparing the next batch. Track the
        # worker that actually performed the final shared-buffer read; a
        # retrieval verify is followed by neural draft-extend during warm-up.
        self._last_shared_buffer_reader = self.neural_worker

        # Per-route scheduling metrics, reset on flush_cache.
        self._route_stats = HybridRouteStats()
        self._runtime_states: dict[str, HybridRuntimeState] = {}
        self._active_runtime_state: HybridRuntimeState | None = None

    @staticmethod
    def _load_config(config_json: str) -> dict[str, Any]:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise ValueError("HYBRID config must be a JSON object.") from exc
        if not isinstance(config, dict):
            raise ValueError("HYBRID config must be a JSON object.")
        for role in ("retrieval", "neural"):
            if not isinstance(config.get(role), dict):
                raise ValueError(f"HYBRID config requires a {role} object.")
        config.setdefault("min_continuation_ratio", 0.25)
        config.setdefault("min_matching_ratio", 0.5)
        for name in ("min_continuation_ratio", "min_matching_ratio"):
            value = config[name]
            if not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError(f"HYBRID {name} must be in (0, 1].")
        return config

    @staticmethod
    def _parse_worker_configs(config: dict[str, Any]) -> dict[str, HybridWorkerConfig]:
        configs = {}
        for role in ("retrieval", "neural"):
            raw = config[role]
            algorithm_name = raw.get("algorithm")
            if not isinstance(algorithm_name, str):
                raise ValueError(f"HYBRID {role}.algorithm must be a string.")
            algorithm = SpeculativeAlgorithm.from_string(algorithm_name)
            if algorithm.is_none() or algorithm.is_hybrid():
                raise ValueError(f"Unsupported nested HYBRID algorithm: {algorithm_name}.")
            overrides = {key: value for key, value in raw.items() if key != "algorithm"}
            configs[role] = HybridWorkerConfig(role, algorithm, overrides)
        return configs

    def _create_workers(self, *worker_args) -> dict[str, BaseSpecWorker]:
        workers = {}
        for role, worker_config in self.worker_configs.items():
            worker_server_args = deepcopy(self.server_args)
            worker_server_args.speculative_algorithm = worker_config.algorithm.name
            for key, value in worker_config.overrides.items():
                if not hasattr(worker_server_args, key):
                    raise ValueError(f"Unknown ServerArgs override for {role}: {key}.")
                setattr(worker_server_args, key, value)
            worker_config.algorithm.handle_server_args(worker_server_args)
            worker_cls = worker_config.algorithm.create_worker(worker_server_args)
            workers[role] = worker_cls(
                server_args=worker_server_args,
                gpu_id=worker_args[0],
                tp_rank=worker_args[1],
                dp_rank=worker_args[2],
                moe_ep_rank=worker_args[3],
                attn_cp_rank=worker_args[4],
                moe_dp_rank=worker_args[5],
                nccl_port=worker_args[6],
                target_worker=self._target_worker,
            )
        return workers

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self.neural_worker.draft_worker

    @property
    def war_fastpath_runner(self):
        return self._last_shared_buffer_reader.war_fastpath_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Scheduler setup needs the union because routes may switch at every
        # batch boundary; workers initialize their own backends up front.
        return tuple(
            dict.fromkeys(
                backend
                for worker in self.workers.values()
                for backend in worker.spec_v2_attn_backends
            )
        )

    def alloc_memory_pool(self, **kwargs):
        for worker in self.workers.values():
            worker.alloc_memory_pool(**kwargs)

    def init_attention_backends(self):
        for worker in self.workers.values():
            worker.init_attention_backends()
        self.neural_worker.init_hybrid_draft_extend_widths(
            self._hybrid_draft_extend_widths
        )

    def init_cuda_graphs(self):
        for worker in self.workers.values():
            worker.init_cuda_graphs()
        self.neural_worker.init_hybrid_target_verify_widths(
            self._hybrid_draft_extend_widths
        )
        self._init_runtime_states()

    def _init_runtime_states(self) -> None:
        """Build complete route states after every backend/graph is initialized."""
        target_runner = self._target_worker.model_runner
        graph_runners = getattr(target_runner, "hybrid_target_verify_graph_runners", {})
        attn_backends = getattr(target_runner, "hybrid_target_verify_attn_backends", {})
        states = {}
        for route, worker in self.workers.items():
            width = worker.speculative_num_draft_tokens
            target_graph_runner = graph_runners.get(width)
            target_attn_backend = attn_backends.get(width)
            if target_attn_backend is None and target_graph_runner is not None:
                target_attn_backend = target_graph_runner.attn_backend
            if target_attn_backend is None:
                target_attn_backend = target_runner.attn_backend
            states[route] = HybridRuntimeState(
                route=route,
                speculative_num_draft_tokens=width,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
            )
        self._runtime_states = states
        self._apply_runtime_state(self._last_route)

    def _apply_runtime_state(self, route: str) -> None:
        """Atomically switch target verify resources at a route boundary.

        Like ``EAGLEWorkerV2.apply_runtime_state``, never combine a backend
        from one width with the graph captured for another width.  The helper
        is deliberately invoked before draft preparation so all later
        metadata writes and target replay use this same state.
        """
        state = self._runtime_states.get(route)
        if state is None:
            raise RuntimeError(f"HYBRID runtime state is missing for route={route}.")
        target_runner = self._target_worker.model_runner
        # The eager TARGET_VERIFY path derives shape metadata from ServerArgs,
        # whereas a graph runner already owns its capture width. Keep this
        # scalar synchronized in both modes so a neural width never reuses the
        # retrieval bootstrap width after a route switch.
        #
        # Do not short-circuit this write merely because the logical route is
        # unchanged. The other child worker may have installed its own target
        # backend / graph while warming runtime state in the previous overlap
        # iteration. Reasserting the whole route state at every batch boundary
        # makes the verify input, attention metadata, and graph replay agree.
        target_runner.server_args.speculative_num_draft_tokens = (
            state.speculative_num_draft_tokens
        )
        target_runner.attn_backend = state.target_attn_backend
        target_runner.decode_cuda_graph_runner = state.target_graph_runner
        self._active_runtime_state = state

    def clear_cache_pool(self):
        for worker in self.workers.values():
            worker.clear_cache_pool()
        self._route_stats.reset()

    def on_verify_complete_cpu(self, num_correct_drafts_per_req, batch_size=0):
        # Verification statistics belong to the route that produced them. In
        # particular, feeding NGRAM acceptance into neural adaptive tuning
        # would corrupt its step selection after a route switch.
        self.workers[self._last_route].on_verify_complete_cpu(
            num_correct_drafts_per_req, batch_size=batch_size
        )

    def activate_step_by_batch(self, batch_size: int):
        # Give each role a batch-boundary activation opportunity. Retrieval is
        # currently static, while the neural role owns draft steps; calling both
        # is necessary for future retrieval/adaptive workers and avoids keeping
        # inactive route state stale.
        for worker in self.workers.values():
            worker.activate_step_by_batch(batch_size)
        # ``speculative_num_steps`` is the controller's scheduler-facing value
        # for the neural draft phase. Retrieval has no draft-model steps.
        self.speculative_num_steps = self.neural_worker.speculative_num_steps
        self._accept_width = max(
            self.retrieval_worker.speculative_num_draft_tokens,
            self.neural_worker.speculative_num_draft_tokens,
        )
        self.speculative_num_draft_tokens = self._accept_width

    def _match_continuation_lengths(self, batch: ScheduleBatch) -> list[int]:
        """Measure the current suffix's actual retrieval continuation.

        The retrieval worker performs the same lookup it will use for drafting;
        the controller only consumes its per-request continuation lengths.
        """
        return self.retrieval_worker.get_retrieval_continuation_lengths(batch)

    def _should_use_retrieval(
        self, batch: ScheduleBatch, lengths: list[int]
    ) -> bool:
        if not batch.forward_mode.is_decode() or batch.is_extend_in_batch:
            return False
        # Root occupies one verify slot. Convert each ratio at the current
        # retrieval width / batch size so routing stays stable as batches vary.
        min_continuation = math.ceil(
            (self.retrieval_worker.speculative_num_draft_tokens - 1)
            * self.routing["min_continuation_ratio"]
        )
        qualifying = sum(length >= min_continuation for length in lengths)
        min_matching = math.ceil(
            len(lengths) * self.routing["min_matching_ratio"]
        )
        return qualifying >= min_matching

    def _update_neural_runtime_state(self, batch: ScheduleBatch, result) -> None:
        source_draft_token_num = result.speculative_num_draft_tokens
        accept_buffer_width = getattr(
            result.next_draft_input,
            "accept_buffer_width",
            source_draft_token_num,
        )
        accept_tokens = result.next_draft_input.accept_tokens.view(
            len(batch.reqs), accept_buffer_width
        )[:, :source_draft_token_num]
        bonus_indices = (result.accept_lens - 1).to(torch.long).unsqueeze(1)
        bonus_tokens = accept_tokens.gather(1, bonus_indices).squeeze(1)
        self.neural_worker.update_runtime_state_from_hybrid(
            batch,
            result,
            bonus_tokens,
            source_draft_token_num=source_draft_token_num,
        )

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        original_spec_info = batch.spec_info
        is_pure_decode = batch.forward_mode.is_decode() and not batch.is_extend_in_batch
        if is_pure_decode:
            lengths = self._match_continuation_lengths(batch)
        else:
            lengths = []
        if self._should_use_retrieval(batch, lengths):
            self._apply_runtime_state("retrieval")
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            result = self.retrieval_worker.forward_batch_generation(batch, on_publish)
            self._update_neural_runtime_state(batch, result)
            last_shared_buffer_reader = self.neural_worker
            route = "retrieval"
        else:
            self._apply_runtime_state("neural")
            batch.spec_info = original_spec_info
            result = self.neural_worker.forward_batch_generation(batch, on_publish)
            self.retrieval_worker.update_runtime_state_from_hybrid(batch, result)
            last_shared_buffer_reader = self.neural_worker
            route = "neural"
        self._last_shared_buffer_reader = last_shared_buffer_reader
        self._update_route_stats(route, result, lengths, is_pure_decode)
        if route != self._last_route:
            logger.debug("Hybrid speculative route switched to %s", route)
            self._last_route = route
        return result

    def _update_route_stats(
        self,
        route: str,
        result,
        continuation_lengths: list[int],
        is_pure_decode: bool,
    ) -> None:
        """Accumulate per-route metrics and optionally emit a debug log."""
        stats = self._route_stats
        bs = len(result.accept_lens) if result.accept_lens is not None else 0
        accept_lens = result.accept_lens

        if is_pure_decode:
            if route == "retrieval":
                stats.retrieval_forward_ct += 1
                if bs > 0 and accept_lens is not None:
                    stats.retrieval_num_accept_tokens += float(accept_lens.float().mean().item())
                    stats.retrieval_num_draft_tokens += (
                        self.retrieval_worker.speculative_num_draft_tokens - 1
                    )
            else:
                stats.neural_forward_ct += 1
                if bs > 0 and accept_lens is not None:
                    stats.neural_num_accept_tokens += float(accept_lens.float().mean().item())
                    stats.neural_num_draft_tokens += (
                        self.neural_worker.speculative_num_draft_tokens - 1
                    )

            min_continuation = math.ceil(
                (self.retrieval_worker.speculative_num_draft_tokens - 1)
                * self.routing["min_continuation_ratio"]
            )
            stats.total_continuation_length += (
                sum(continuation_lengths) / len(continuation_lengths)
                if continuation_lengths
                else 0.0
            )
            stats.matching_req_ct += sum(l >= min_continuation for l in continuation_lengths)

        log_interval = envs.SGLANG_LOG_HYBRID_SPEC.get()
        if (
            is_pure_decode
            and log_interval > 0
            and stats.total_forward_ct % log_interval == 0
        ):
            avg_cont = stats.avg_continuation_length
            min_cont = math.ceil(
                (self.retrieval_worker.speculative_num_draft_tokens - 1)
                * self.routing["min_continuation_ratio"]
            )
            cont_info = (
                f"cont_len={continuation_lengths} "
                f"min_cont={min_cont} "
                if continuation_lengths
                else ""
            )
            logger.info(
                "HybridSpec route=%s bs=%d %s"
                "avg_cont=%.2f matching_req_ct=%d | "
                "retrieval: forward_ct=%d accept_len=%.2f accept_rate=%.2f | "
                "neural: forward_ct=%d accept_len=%.2f accept_rate=%.2f",
                route,
                bs,
                cont_info,
                avg_cont,
                stats.matching_req_ct,
                stats.retrieval_forward_ct,
                stats.retrieval_accept_length,
                stats.retrieval_accept_rate,
                stats.neural_forward_ct,
                stats.neural_accept_length,
                stats.neural_accept_rate,
            )
