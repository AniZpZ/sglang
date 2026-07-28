from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import torch

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
        for field in ("min_continuation_ratio", "min_matching_ratio"):
            value = config[field]
            if not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError(f"HYBRID {field} must be in (0, 1].")
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

    def clear_cache_pool(self):
        for worker in self.workers.values():
            worker.clear_cache_pool()

    def on_verify_complete_cpu(self, num_correct_drafts_per_req, batch_size=0):
        self.neural_worker.on_verify_complete_cpu(
            num_correct_drafts_per_req, batch_size=batch_size
        )

    def activate_step_by_batch(self, batch_size: int):
        self.neural_worker.activate_step_by_batch(batch_size)
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
        accept_tokens = result.next_draft_input.accept_tokens.view(
            len(batch.reqs), source_draft_token_num
        )
        bonus_indices = (result.accept_lens - 1).to(torch.long).unsqueeze(1)
        bonus_tokens = accept_tokens.gather(1, bonus_indices).squeeze(1)
        self.neural_worker.update_runtime_state_from_hybrid(
            batch,
            result,
            bonus_tokens,
            source_draft_token_num=source_draft_token_num,
        )

    def _attach_accept_path(self, batch: ScheduleBatch, result) -> None:
        """Attach a fixed-width accepted-path relay for the next route decision."""
        width = result.speculative_num_draft_tokens or self._accept_width
        if result.accept_lens is None:
            result.accept_lens = torch.ones(
                len(batch.reqs), dtype=torch.int32, device=result.next_token_ids.device
            )
        worker_accept_tokens = getattr(result.next_draft_input, "accept_tokens", None)
        if worker_accept_tokens is None:
            if result.next_token_ids.numel() == len(batch.reqs) * width:
                accept_tokens = result.next_token_ids.view(len(batch.reqs), width)
            else:
                accept_tokens = torch.zeros(
                    len(batch.reqs),
                    width,
                    dtype=result.next_token_ids.dtype,
                    device=result.next_token_ids.device,
                )
                accept_tokens[:, 0] = result.next_token_ids.reshape(len(batch.reqs))
        else:
            accept_tokens = worker_accept_tokens.view(len(batch.reqs), width)
        relay_tokens = torch.zeros(
            len(batch.reqs),
            self._accept_width,
            dtype=accept_tokens.dtype,
            device=accept_tokens.device,
        )
        relay_tokens[:, :width] = accept_tokens
        result.next_draft_input.accept_tokens = relay_tokens
        result.next_draft_input.accept_lens = result.accept_lens
        result.next_draft_input.accept_width = self._accept_width

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        original_spec_info = batch.spec_info
        if batch.forward_mode.is_decode() and not batch.is_extend_in_batch:
            lengths = self._match_continuation_lengths(batch)
        else:
            lengths = []
        if self._should_use_retrieval(batch, lengths):
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            result = self.retrieval_worker.forward_batch_generation(batch, on_publish)
            self._update_neural_runtime_state(batch, result)
            last_shared_buffer_reader = self.neural_worker
            route = "retrieval"
        else:
            batch.spec_info = original_spec_info
            result = self.neural_worker.forward_batch_generation(batch, on_publish)
            self.retrieval_worker.update_runtime_state_from_hybrid(batch, result)
            last_shared_buffer_reader = self.neural_worker
            route = "neural"
        self._attach_accept_path(batch, result)
        self._last_shared_buffer_reader = last_shared_buffer_reader
        if route != self._last_route:
            logger.debug("Hybrid speculative route switched to %s", route)
            self._last_route = route
        return result
