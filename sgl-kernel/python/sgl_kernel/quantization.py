import torch

from sgl_kernel.scalar_type import ScalarType

from typing import Optional

def gptq_marlin_gemm(a: torch.Tensor,
                     c: Optional[torch.Tensor],
                     b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor,
                     global_scale: Optional[torch.Tensor],
                     b_zeros: Optional[torch.Tensor],
                     g_idx: Optional[torch.Tensor],
                     perm: Optional[torch.Tensor],
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool = True,
                     use_atomic_add: bool = False,
                     use_fp32_reduce: bool = False,
                     is_zp_float: bool = False) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_marlin_gemm(a, c, b_q_weight, b_scales,
                                                 global_scale, b_zeros, g_idx, perm,
                                                 workspace, b_q_type.id, size_m,
                                                 size_n, size_k, is_k_full,
                                                 use_atomic_add, use_fp32_reduce,
                                                 is_zp_float)