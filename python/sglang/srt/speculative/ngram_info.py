from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.speculative.eagle_info_v2 import EagleDraftInputV2Mixin
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


class NgramVerifyInput(SpecInput, EagleDraftInputV2Mixin):
    def __init__(
        self,
        draft_token: torch.Tensor = None,
        custom_mask: torch.Tensor = None,
        positions: torch.Tensor = None,
        retrieve_index: torch.Tensor = None,
        retrieve_next_token: torch.Tensor = None,
        retrieve_next_sibling: torch.Tensor = None,
        draft_token_num: int = None,
        grammar: BaseGrammarObject = None,
        future_indices: Optional[torch.Tensor] = None,
        new_seq_lens: Optional[torch.Tensor] = None,
        accept_tokens: Optional[torch.Tensor] = None,
        accept_lens: Optional[torch.Tensor] = None,
        bonus_tokens: Optional[torch.Tensor] = None,
        accept_width: Optional[int] = None,
        accept_buffer_width: Optional[int] = None,
        num_tokens_per_req: Optional[int] = None,
    ):
        super().__init__(SpecInputType.NGRAM_VERIFY)
        self.draft_token = draft_token
        self.custom_mask = custom_mask
        self.positions = positions
        self.retrieve_index = retrieve_index
        self.retrieve_next_token = retrieve_next_token
        self.retrieve_next_sibling = retrieve_next_sibling
        self.draft_token_num = draft_token_num
        # ForwardBatch uses this generic protocol field to reconstruct the
        # request batch size for target-verify forwards.
        self.num_tokens_per_req = (
            draft_token_num if num_tokens_per_req is None else num_tokens_per_req
        )
        self.grammar = grammar

        # Inputs for V2 overlap worker
        self.future_indices = future_indices
        self.new_seq_lens = new_seq_lens
        self.accept_tokens = accept_tokens
        self.accept_lens = accept_lens
        # NGRAM consumes the complete accepted path on CPU. ``bonus_tokens``
        # is separately retained for EAGLE's GPU draft-state relay.
        self.bonus_tokens = bonus_tokens
        self.accept_width = (
            draft_token_num if accept_width is None else accept_width
        )
        self.accept_buffer_width = (
            self.accept_width if accept_buffer_width is None else accept_buffer_width
        )
        self.device = (
            custom_mask.device if custom_mask is not None else new_seq_lens.device
        )

    @property
    def max_tree_depth(self) -> int:
        # NGRAM trees are node-budgeted with no depth cap: the corpus BFS only
        # stops on the node budget, so a single long match can chain all
        # draft_token_num nodes (spec_steps is meaningless for this tree).
        return self.draft_token_num

    @property
    def tree_topk(self) -> int:
        # Irregular tree: per-level branching follows the corpus matches.
        return -1

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        bs = len(req_pool_indices)

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        self.qo_indptr = (
            torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
            * self.draft_token_num
        )

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=self.device,
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )

        # Pad custom_mask when CUDA graph pads batch size beyond the actual number of requests.
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * bs
        )
        custom_mask = self.custom_mask
        if custom_mask.numel() < mask_numel:
            custom_mask = torch.cat(
                [
                    custom_mask,
                    torch.full(
                        (mask_numel - custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, self.qo_indptr, custom_mask

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
        if self.new_seq_lens is not None:
            self.new_seq_lens = self.new_seq_lens[new_indices]
        # Hybrid preserves producer width separately from its padded relay
        # stride. Filter by the latter so a source-width-4 path padded to 6
        # never shifts following request rows.
        buffer_width = self.accept_buffer_width or self.draft_token_num
        self.accept_tokens = self.accept_tokens.reshape(-1, buffer_width)[
            new_indices,
            :
        ]
        self.accept_tokens = self.accept_tokens.flatten()
        self.accept_lens = self.accept_lens[new_indices]
        if self.bonus_tokens is not None:
            self.bonus_tokens = self.bonus_tokens[new_indices]
        if isinstance(self.accept_width, torch.Tensor):
            self.accept_width = self.accept_width[new_indices]

    def merge_batch(self, spec_info: NgramVerifyInput):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                (self.future_indices, spec_info.future_indices), dim=0
            )
        if self.new_seq_lens is not None:
            assert spec_info.new_seq_lens is not None
            self.new_seq_lens = torch.cat(
                (self.new_seq_lens, spec_info.new_seq_lens), dim=0
            )
        # Pad narrower side to the wider relay stride before concatenation so
        # the resulting tensor has a consistent row layout.
        self_width = self.accept_buffer_width or self.accept_width or self.draft_token_num
        other_width = (
            spec_info.accept_buffer_width
            or spec_info.accept_width
            or spec_info.draft_token_num
        )
        merged_width = max(self_width, other_width)
        self_2d = self.accept_tokens.reshape(-1, self_width)
        other_2d = spec_info.accept_tokens.reshape(-1, other_width)
        if self_width < merged_width:
            self_2d = torch.nn.functional.pad(
                self_2d, (0, merged_width - self_width)
            )
        if other_width < merged_width:
            other_2d = torch.nn.functional.pad(
                other_2d, (0, merged_width - other_width)
            )
        self.accept_tokens = torch.cat((self_2d, other_2d), dim=0).flatten()
        self.accept_buffer_width = merged_width
        self.accept_lens = torch.cat((self.accept_lens, spec_info.accept_lens), dim=0)
        if self.bonus_tokens is not None:
            assert spec_info.bonus_tokens is not None
            self.bonus_tokens = torch.cat(
                (self.bonus_tokens, spec_info.bonus_tokens), dim=0
            )
        else:
            assert spec_info.bonus_tokens is None
        self_source_width = torch.as_tensor(
            self.accept_width, dtype=torch.int32, device=self.accept_lens.device
        ).expand(self_2d.shape[0])
        other_source_width = torch.as_tensor(
            spec_info.accept_width,
            dtype=torch.int32,
            device=self.accept_lens.device,
        ).expand(other_2d.shape[0])
        self.accept_width = torch.cat((self_source_width, other_source_width), dim=0)
