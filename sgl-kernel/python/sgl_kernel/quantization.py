import torch

# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                          b_g_idx, use_exllama, bit)