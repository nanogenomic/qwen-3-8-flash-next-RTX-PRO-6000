# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
# Build a patched copy of flashinfer 0.6.18 cutlass_fused_moe_kernels.cuh whose finalizeMoeRouting launch
# uses an ILP/column-split variant with IDENTICAL per-element arithmetic.
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

kernel = r'''
// ---- qwen-opt: latency-bound finalize, same math, more memory-level parallelism.
// The stock kernel walks the top-k list serially (3 dependent global loads per k) with one CTA per token.
// This variant (a) gathers the k routing metadata once per thread, (b) issues all k row loads before any
// accumulation, (c) splits the columns over gridDim.y. Per output element the fp32 accumulation is the
// same expression in the same k order, so the result is bitwise identical to finalizeMoeRoutingKernel.
constexpr static int QWENOPT_FINALIZE_MAX_K = 16;
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingIlpKernel(
    GemmOutputType const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias, float const* scales, int const* unpermuted_row_to_permuted_row,
    int const* token_selected_experts, int64_t const num_rows, int64_t const padded_cols,
    int64_t const unpadded_cols, int64_t const experts_per_token, int const num_experts_per_node,
    int const start_expert_id) {
  int64_t const original_row = blockIdx.x;
  auto const offset = original_row * unpadded_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;
  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);
  int64_t const start_offset = threadIdx.x + static_cast<int64_t>(blockIdx.y) * blockDim.x;
  int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.y;
  int64_t const num_elems_in_padded_col = padded_cols / FINALIZE_ELEM_PER_THREAD;
  int64_t const num_elems_in_orig_col = unpadded_cols / FINALIZE_ELEM_PER_THREAD;
  using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
  using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
  auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int64_t k_perm[QWENOPT_FINALIZE_MAX_K];
  int64_t k_expert[QWENOPT_FINALIZE_MAX_K];
  float k_scale[QWENOPT_FINALIZE_MAX_K];
  bool k_ok[QWENOPT_FINALIZE_MAX_K];
  int64_t const expanded_rows = num_rows * experts_per_token;
#pragma unroll
  for (int k_idx = 0; k_idx < QWENOPT_FINALIZE_MAX_K; ++k_idx) {
    k_ok[k_idx] = false;
    k_perm[k_idx] = 0;
    k_expert[k_idx] = 0;
    k_scale[k_idx] = 1.f;
    if (k_idx < experts_per_token) {
      int64_t const k_offset = original_row * experts_per_token + k_idx;
      int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;
      if (expert_id >= 0 && expert_id < num_experts_per_node) {
        int64_t const expanded_original_row = original_row + k_idx * num_rows;
        int64_t const expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];
        if (expanded_permuted_row >= 0 && expanded_permuted_row < expanded_rows) {
          k_ok[k_idx] = true;
          k_perm[k_idx] = expanded_permuted_row;
          k_expert[k_idx] = expert_id;
          k_scale[k_idx] = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
        }
      }
    }
  }

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_orig_col; elem_index += stride) {
    InputElem k_val[QWENOPT_FINALIZE_MAX_K];
#pragma unroll
    for (int k_idx = 0; k_idx < QWENOPT_FINALIZE_MAX_K; ++k_idx) {
      if (k_ok[k_idx]) {
        k_val[k_idx] = expanded_permuted_rows_v[k_perm[k_idx] * num_elems_in_padded_col + elem_index];
      }
    }
    ComputeElem thread_output;
    thread_output.fill(0);
#pragma unroll
    for (int k_idx = 0; k_idx < QWENOPT_FINALIZE_MAX_K; ++k_idx) {
      if (k_ok[k_idx]) {
        float const row_scale = k_scale[k_idx];
        ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(k_val[k_idx]);
        if (bias) {
          auto const* bias_ptr = bias_v + k_expert[k_idx] * num_elems_in_padded_col;
          expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
        }
        thread_output = thread_output + row_scale * expert_result;
      }
    }
    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}
'''
anchor = "// Final kernel to unpermute and scale\n// This kernel unpermutes the original data, does the k-way reduction and performs the final skip\n// connection.\ntemplate <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>\n__global__ void finalizeMoeRoutingNoFillingKernel("
assert s.count(anchor) == 1, "anchor"
s = s.replace(anchor, kernel + "\n" + anchor, 1)

old_launch = '''    int64_t const blocks = num_rows;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
    config.blockDim = threads;
    auto func = final_scales ? &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::DEFAULT>
                             : &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::NO_SCALE>;
    cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr,
                       final_scales, unpermuted_row_to_permuted_row, token_selected_experts,
                       padded_cols, unpadded_cols, experts_per_token, num_experts_per_node,
                       start_expert_id);'''
new_launch = '''    if (experts_per_token <= QWENOPT_FINALIZE_MAX_K) {
      // qwen-opt: one CTA per (token, column slice); 64 threads x 8 elems -> 512 cols per slice.
      constexpr int64_t EPT =
          128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);
      int64_t const threads = 64;
      int64_t const vec_cols = unpadded_cols / EPT;
      int64_t const slices = std::max<int64_t>(1, (vec_cols + threads - 1) / threads);
      config.gridDim = dim3(static_cast<unsigned>(num_rows), static_cast<unsigned>(slices), 1);
      config.blockDim = threads;
      auto func = final_scales ? &finalizeMoeRoutingIlpKernel<OutputType, GemmOutputType,
                                                              ScaleBiasType, ScaleMode::DEFAULT>
                               : &finalizeMoeRoutingIlpKernel<OutputType, GemmOutputType,
                                                              ScaleBiasType, ScaleMode::NO_SCALE>;
      cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr,
                         final_scales, unpermuted_row_to_permuted_row, token_selected_experts,
                         num_rows, padded_cols, unpadded_cols, experts_per_token,
                         num_experts_per_node, start_expert_id);
      return;
    }
    int64_t const blocks = num_rows;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
    config.blockDim = threads;
    auto func = final_scales ? &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::DEFAULT>
                             : &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::NO_SCALE>;
    cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr,
                       final_scales, unpermuted_row_to_permuted_row, token_selected_experts,
                       padded_cols, unpadded_cols, experts_per_token, num_experts_per_node,
                       start_expert_id);'''
assert s.count(old_launch) == 1, "launch"
s = s.replace(old_launch, new_launch, 1)
open(dst, "w").write(s)
print("patched ok")
