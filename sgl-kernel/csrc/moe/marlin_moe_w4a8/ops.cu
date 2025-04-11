/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

 #ifndef MARLIN_W4A8_NAMESPACE_NAME
 #define MARLIN_W4A8_NAMESPACE_NAME marlin_moe_w4a8
#endif

#include "kernel.h"
#include "core/registration.h"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
 static_assert(std::is_same<scalar_t, half>::value ||          \
                   std::is_same<scalar_t, nv_bfloat16>::value, \
               "only float16 and bfloat16 is supported");

namespace MARLIN_W4A8_NAMESPACE_NAME {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800


}  // namespace marlin

torch::Tensor moe_w4a8_marlin_gemm(
   torch::Tensor& a, std::optional<torch::Tensor> const& c_or_none,
   torch::Tensor& b_q_weight, torch::Tensor& b_scales,
   std::optional<torch::Tensor> const& b_zeros_or_none,
   std::optional<torch::Tensor> const& g_idx_or_none,
   std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
   torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
   torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
   int64_t moe_block_size, int64_t top_k, bool mul_topk_weights, bool is_ep,
   vllm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
   int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
   bool is_zp_float) {
 TORCH_CHECK_NOT_IMPLEMENTED(false,
                             "marlin_gemm(..) requires CUDA_ARCH >= 8.0");
 return torch::empty({1, 1});
}

#else


typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},  // Default
    {128, 64, 128},   // Reduce N 2X, same K
    {64, 256, 256},   // Reduce K 2X, increase N 2X
    {64, 128, 128},   // Reduce K 2X, same N
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},   // Default
    {128, 128, 256},  // Reduce N 2X, increase K 2X
    {64, 128, 128},   // Reduce N 2X, same K
    {128, 64, 128},   // Reduce N 4X, increase K 2X
};

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int group_size) {

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  int tb_scales = tb_groups * tb_n * 2;

  return tb_scales * pipe_stages;
}

bool is_valid_cache_size(thread_config_t const& th_config, int moe_block_size,
                         int prob_m, int prob_n, int prob_k,
                         int scales_cache_size, int max_shared_mem) {
  int pack_factor = 8;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;

  int b_size = (tb_k * tb_n / pack_factor) * 4;

  // Get A size
  int tb_max_m = moe_block_size;
  int a_size = (tb_max_m * tb_k) * 2;

  float pipe_size = (a_size + b_size) * pipe_stages;

  float reduce_size = max(th_config.num_threads * 32 * 4,
                          (tb_n / 64) * 32 * (tb_max_m / 16) * 4 * 2 * 4 * 2);

  TORCH_CHECK(max_shared_mem / 2 > scales_cache_size);  // Sanity

  return pipe_size + reduce_size < 0.95f * (max_shared_mem - scales_cache_size);
}

bool is_valid_config(thread_config_t const& th_config, int moe_block_size, int prob_m, int prob_n,
                     int prob_k, int group_size, int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // thread_k can be only 128 or 64 (because it must be less than groupsize
  // which is 128)
  if (th_config.thread_k != 128 && th_config.thread_k != 64) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  //  Determine cache for scales
  int scales_cache_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k,
                            group_size);

  // Check that pipeline fits into cache
  if (!is_valid_cache_size(th_config, moe_block_size, prob_m, prob_n, prob_k,
                           scales_cache_size, max_shared_mem)) {
    return false;
  }

  return true;
}

thread_config_t determine_thread_config(int prob_m, int prob_n, int prob_k
                                        int moe_block_size,int group_size,
                                        int max_shared_mem) {
  if (moe_block_size <= 16) {
    for (auto th_config : small_batch_thread_configs) {
      if (is_valid_config(th_config, moe_block_size, prob_m, prob_n, prob_k,
                          group_size, max_shared_mem)) {
        return th_config;
      }
    }

  } else {
    for (auto th_config : large_batch_thread_configs) {
      if (is_valid_config(th_config, moe_block_size, prob_m, prob_n, prob_k,
                          group_size, max_shared_mem)) {
        return th_config;
      }
    }
  }

  return thread_config_t{-1, -1, -1};
}


#define __CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,               \
    GROUP_BLOCKS, NUM_THREADS)                                       \
else if (thread_m_blocks == THREAD_M_BLOCKS &&                                   \
thread_n_blocks == THREAD_N_BLOCKS &&                                   \
thread_k_blocks == THREAD_K_BLOCKS &&                                   \
group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {           \
cudaFuncSetAttribute(Marlin<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,     \
                  THREAD_K_BLOCKS, pipe_stages, GROUP_BLOCKS>,            \
           cudaFuncAttributeMaxDynamicSharedMemorySize,              \
           max_shared_mem);                                          \
Marlin<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,         \
pipe_stages, GROUP_BLOCKS>                                                   \
<<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                         \
A_ptr, B_ptr, C_ptr, D_ptr, s1_ptr, s2_ptr, s3_ptr,                    \
sorted_token_ids_ptr, expert_ids_ptr,                                  \
num_tokens_past_padded_ptr, topk_weights_ptr, top_k,                   \
mul_topk_weights, is_ep, num_groups,                                   \
prob_m, prob_n, prob_k, locks);                                        \
}

#define CALL_IF(N_BLOCKS, K_BLOCKS, NUM_THREADS)    \
__CALL_IF(1, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS) \
__CALL_IF(1, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)  \
__CALL_IF(1, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS) \
__CALL_IF(1, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)  \
__CALL_IF(2, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS) \
__CALL_IF(2, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)  \
__CALL_IF(3, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS) \
__CALL_IF(3, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)  \
__CALL_IF(4, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS) \
__CALL_IF(4, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

void marlin_w4a8_mm(
    const void* A,
    const void* B,
          void* C, // int32 reduce buffer
          void* D, // half
    const void* s1,
    const void* s2,
    const void* s3,
    void* sorted_token_ids,  //moe start
    void* expert_ids,
    void* num_tokens_past_padded, 
    void* topk_weights,
    int moe_block_size, 
    int top_k, 
    bool mul_topk_weights, 
    bool is_ep,             //moe end
    int prob_m,
    int prob_n,
    int prob_k,
    void* workspace,
    int groupsize = -1,
    bool is_k_full, //optional
    int dev = 0,
    cudaStream_t stream = 0,
    int thread_k = -1,
    int thread_n = -1,
    int sms = -1,
    int max_par = 16
  ) {
    int thread_m_blocks = moe_block_size / 16;
  
    int tot_m = prob_m;
    int tot_m_blocks = ceildiv(tot_m, 16);
    
    // hongbo: whether need it
    int pad = 16 * tot_m_blocks - tot_m;
  
    if (sms == -1)
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  
    // Set thread config
    thread_config_t th_config;
    if (thread_k != -1 && thread_n != -1) {
      // User-defined config
      th_config = thread_config_t{thread_k, thread_n, USER_THREADS};
    } else {
      // Auto config
      th_config = determine_thread_config(prob_m, prob_n, prob_k, moe_block_size,
                                          group_size, max_shared_mem);
    }
    // int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
    int group_blocks = 0;
    if (group_size == -1) {
        group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                    " is not divisible by group_blocks = ", group_blocks);
    }
    
    if (!is_valid_config(th_config, prob_m, prob_n, prob_k) || (group_blocks != -1 && prob_k % group_blocks != 0))
      return ERR_PROB_SHAPE;
    TORCH_CHECK(!is_valid_config(th_config, prob_m, prob_n, prob_k) || (group_blocks != -1 && prob_k % group_blocks != 0),
                "Invalid thread config: moe_block_size = ", 0,
                ", thread_k = ", th_config.thread_k,
                ", thread_n = ", th_config.thread_n,
                ", num_threads = ", th_config.num_threads, " for MKN = [",
                prob_m, ", ", prob_k, ", ", prob_n, "] and num_bits = ", 4,
                ", group_size = ", group_size,);
    
    int num_threads = th_config.num_threads;
    thread_k = th_config.thread_k;
    thread_n = th_config.thread_n;
    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;
    int blocks = sms;
  
    if (groupsize == -1)
      assert(s3 == nullptr);
    if (prob_m == 0 || prob_n == 0 || prob_k == 0)
      return 0;
    TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                " is not divisible by thread_n = ", thread_n);
    TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                " is not divisible by thread_k = ", thread_k);
  
    const int4* A_ptr = (const int4*) A;
    const int4* B_ptr = (const int4*) B;
    int4* C_ptr = (int4*) C;
    int4* D_ptr = (int4*) D;
    const float* s1_ptr = (const float*) s1;
    const int4* s2_ptr = (const int4*) s2;
    const int4* s3_ptr = (const int4*) s3;
    // moe param
    const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
    const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
    const int32_t* num_tokens_past_padded_ptr =
        (const int32_t*)num_tokens_past_padded;
    const float* topk_weights_ptr = (const float*)topk_weights;
    // moe param
    int* locks = (int*) workspace;
  
    if (false) {}
      CALL_IF(8, 8, 256)
      CALL_IF(16, 4, 256)
      CALL_IF(8, 4, 128)
      CALL_IF(4, 8, 128)
    else {
      TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                  ", ", prob_k, "]",  ", group_size = ", group_size,
                  ", thread_m_blocks = ", thread_m_blocks,
                  ", thread_n_blocks = ", thread_n_blocks,
                  ", thread_k_blocks = ", thread_k_blocks);
    }
  
  }  

}  // namespace MARLIN_W4A8_NAMESPACE_NAME

torch::Tensor moe_w4a8_marlin_gemm(
    torch::Tensor& a, torch::Tensor& a_scale,
    std::optional<torch::Tensor> const& c_or_none,
    torch::Tensor& b_q_weight, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights, bool is_ep,
    sglang::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  sglang::ScalarType const b_q_type = sglang::ScalarType::from_id(b_q_type_id);
  int pack_factor = 32 / b_q_type.size_bits();

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(1),
              "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
              ", size_k = ", size_k,
              ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(
      b_q_weight.size(2) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(2) = ", b_q_weight.size(2),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(2) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.get_device());

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    TORCH_CHECK(c.size(0) == size_m * top_k,
                "Shape mismatch: c.size(0) = ", c.size(0),
                ", size_m * topk = ", size_m * top_k);
    TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                ", size_n = ", size_n);
  } else {
    c = torch::empty({size_m * top_k, size_n}, options);
  }

  // Alloc C tmp buffer that is going to be used for the global reduce
  torch::Tensor c_tmp;
  auto options_fp32 =
      torch::TensorOptions().dtype(at::kFloat).device(a.device());
  if (use_fp32_reduce && !use_atomic_add) {
    const long max_c_tmp_size =
        min(((long)size_n * sorted_token_ids.size(0)),
            (long)(sms * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n));
    c_tmp = torch::empty({max_c_tmp_size}, options_fp32);
  } else {
    c_tmp = torch::empty({0}, options_fp32);
  }

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.sizes().size();
  TORCH_CHECK(rank == 3, "b_scales rank = ", rank, " is not 3");
  TORCH_CHECK(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(1);

  torch::Tensor g_idx, perm, a_tmp;
  ;
  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm = perm_or_none.value();

    TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
    TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");
    TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

    // Verify g_idx and perm
    TORCH_CHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                    (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                " and perm.size(-1) = ", perm.size(-1),
                ", where size_k = ", size_k);
  } else {
    g_idx = torch::empty({0}, options);
    perm = torch::empty({0}, options);
    a_tmp = torch::empty({0}, options);
  }
  bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

  if (has_act_order) {
    a_tmp = torch::empty({size_m * top_k, size_k}, options);
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    a_tmp = torch::empty({0}, options);
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(1) = ", b_scales.size(1));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  // Verify workspace size
  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
              "size_n = ", size_n, ", is not divisible by min_thread_n = ",
              MARLIN_NAMESPACE_NAME::min_thread_n);

  int max_n_tiles = size_n / MARLIN_NAMESPACE_NAME::min_thread_n;
  int min_workspace_size =
      min(max_n_tiles * (int)(sorted_token_ids.size(0) / moe_block_size), sms);
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();
  marlin_qqq_cuda(
      a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), d.data_ptr(),
      s_tok.data_ptr(), s_ch.data_ptr(), s_group.data_ptr(), size_m, size_n,
      size_k, workspace.data_ptr(), groupsize, dev,
      at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms, max_par);

  return c;
}

#endif