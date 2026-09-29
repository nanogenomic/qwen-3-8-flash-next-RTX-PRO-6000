// Copyright © 2025 Ligandal, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// qwen-opt fuse: HyperConnection "tail" kernel.
//
// One CTA per (token row m, branch c), kGroupSize = kHiddenSize elements, laid out
// and reduced exactly like grouped_gemma_rmsnorm_kernel so that the fused RMSNorm is
// bit-identical to the unfused one given the same input:
//
//   [kShared]  y[m, i] = Float( sigmoid(dot(h[m], w_sg)) * s[m, i] + f[m, i] )
//                        (the _fused_gate_sigmoid_mul_add epilogue of the MoE block)
//              else y = block_output
//   a[m, c]    = 2 / (1 + exp(-logit[m, c] / HC))                  (as hc_combine_apply)
//   hyper[m, c*H + i] = Float( r[m, c*H + i] + a[m, c] * y[m, i] ) (as hc_combine_apply)
//   [kNorm]    normed[m, c*H + i] = Float( hyper * rsqrt(mean(hyper^2) + eps) * (1 + w) )
//                                                                  (as grouped_gemma_rmsnorm)
//
// It replaces hc_combine_gate (gate logits come from the mix, hc_mix.fused_hc_mix_gate),
// hc_combine_apply, the next sublayer's grouped_gemma_rmsnorm and, on the MoE side,
// _fused_gate_sigmoid_mul_add.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

struct HcFusedTailParams {
  const void* y;          // [M, H]  block_output, or routed output f when kShared
  const void* s;          // [M, H]  ungated shared-expert output (kShared)
  const void* h;          // [M, H]  MoE input, shared gate operand (kShared)
  const void* w_sg;       // [H]     shared_expert_gate weight (kShared)
  const void* residual;   // [M, HC*H]
  const float* logits;    // [M, HC] combine-gate logits (fp32)
  const void* norm_w;     // [HC*H]  next sublayer hc_norm weight (kNorm)
  void* out_hyper;        // [M, HC*H]
  void* out_normed;       // [M, HC*H] (kNorm)
  float eps;
};

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float, bool kShared, bool kNorm>
__global__ __launch_bounds__(kHiddenSize / 16) void hc_fused_tail_kernel(const HcFusedTailParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
#if SGL_ARCH_BLACKWELL_OR_GREATER
  using Storage = AlignedVector<Float2, 8>;
  constexpr uint32_t kNumLoads = 1;
#else
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kNumLoads = 2;
#endif
  constexpr uint32_t kVecLen = kNumLoads == 1 ? 8 : 4;
  constexpr int64_t kGroupSize = kHiddenSize;
  constexpr auto kNumThreads = kGroupSize / 16;
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;

  const uint32_t bid = blockIdx.x;  // = m * kHcCount + c, same chunk order as grouped rmsnorm
  const uint32_t m = bid / kHcCount;
  const uint32_t c = bid % kHcCount;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[kWarpThreads];
  __shared__ float smem_sg[kWarpThreads];

  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(bid) * kGroupSize);
  const auto y_ptr = pointer::offset<Float>(params.y, static_cast<int64_t>(m) * kHiddenSize);
  const auto out_ptr = pointer::offset<Float>(params.out_hyper, static_cast<int64_t>(bid) * kGroupSize);

  // Static operands (weights) do not depend on the upstream grid.
  Storage norm_w_vec[kNumLoads];
  Storage wsg_vec[kNumLoads];
  if constexpr (kNorm) {
    const auto nw_ptr = pointer::offset<Float>(params.norm_w, static_cast<int64_t>(c) * kGroupSize);
#pragma unroll
    for (uint32_t j = 0; j < kNumLoads; ++j) norm_w_vec[j] = gmem.load(nw_ptr, j);
  }
  if constexpr (kShared) {
#pragma unroll
    for (uint32_t j = 0; j < kNumLoads; ++j) wsg_vec[j] = gmem.load(params.w_sg, j);
  }

  PDLWaitPrimary<kUsePDL>();

  const float total = params.logits[static_cast<int64_t>(m) * kHcCount + c];
  const float a = 2.0f / (1.0f + math::exp(-total / kHcCount));

  Storage y_vec[kNumLoads];
  Storage r_vec[kNumLoads];
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
    r_vec[j] = gmem.load(r_ptr, j);
    y_vec[j] = gmem.load(y_ptr, j);
  }

  if constexpr (kShared) {
    const auto s_ptr = pointer::offset<Float>(params.s, static_cast<int64_t>(m) * kHiddenSize);
    const auto h_ptr = pointer::offset<Float>(params.h, static_cast<int64_t>(m) * kHiddenSize);
    Storage s_vec[kNumLoads];
    Storage h_vec[kNumLoads];
#pragma unroll
    for (uint32_t j = 0; j < kNumLoads; ++j) {
      s_vec[j] = gmem.load(s_ptr, j);
      h_vec[j] = gmem.load(h_ptr, j);
    }
    float dot = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kNumLoads; ++j) {
#pragma unroll
      for (uint32_t i = 0; i < kVecLen; ++i) {
        const auto [hx, hy] = cast<fp32x2_t>(h_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(wsg_vec[j][i]);
        dot += hx * wx + hy * wy;
      }
    }
    dot = warp::reduce_sum(dot);
    const auto warp_id = threadIdx.x / kWarpThreads;
    if constexpr (kNumWarps > 1) {
      if (threadIdx.x % kWarpThreads == 0) smem_sg[warp_id] = dot;
      __syncthreads();
      if (warp_id == 0) {
        const auto tx = threadIdx.x;
        const auto local = tx < kNumWarps ? smem_sg[tx] : 0.0f;
        smem_sg[tx] = warp::reduce_sum(local);
      }
      __syncthreads();
      dot = smem_sg[0];
    }
    const float sg = 1.0f / (1.0f + math::exp(-dot));
#pragma unroll
    for (uint32_t j = 0; j < kNumLoads; ++j) {
#pragma unroll
      for (uint32_t i = 0; i < kVecLen; ++i) {
        const auto [fx, fy] = cast<fp32x2_t>(y_vec[j][i]);
        const auto [sx, sy] = cast<fp32x2_t>(s_vec[j][i]);
        y_vec[j][i] = cast<Float2>(fp32x2_t{sg * sx + fx, sg * sy + fy});
      }
    }
  }

  Storage hyper_vec[kNumLoads];
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
#pragma unroll
    for (uint32_t i = 0; i < kVecLen; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[j][i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[j][i]);
      hyper_vec[j][i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    gmem.store(out_ptr, hyper_vec[j], j);
  }

  if constexpr (kNorm) {
    const auto normed_ptr = pointer::offset<Float>(params.out_normed, static_cast<int64_t>(bid) * kGroupSize);
    float sum_of_squares = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kNumLoads; ++j) {
#pragma unroll
      for (uint32_t i = 0; i < kVecLen; ++i) {
        const auto [x, y] = cast<fp32x2_t>(hyper_vec[j][i]);
        sum_of_squares += x * x + y * y;
      }
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    float norm_factor;
    if constexpr (kNumWarps == 1) {
      norm_factor = math::rsqrt(sum_of_squares / kGroupSize + params.eps);
    } else {
      const auto warp_id = threadIdx.x / kWarpThreads;
      smem[warp_id] = sum_of_squares;
      __syncthreads();
      if (warp_id == 0) {
        const auto tx = threadIdx.x;
        const auto local_sum = tx < kNumWarps ? smem[tx] : 0.0f;
        sum_of_squares = warp::reduce_sum(local_sum);
        smem[tx] = math::rsqrt(sum_of_squares / kGroupSize + params.eps);
      }
      __syncthreads();
      norm_factor = smem[warp_id];
    }

#pragma unroll
    for (uint32_t j = 0; j < kNumLoads; ++j) {
      Storage output_vec;
#pragma unroll
      for (uint32_t i = 0; i < kVecLen; ++i) {
        const auto [ix, iy] = cast<fp32x2_t>(hyper_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(norm_w_vec[j][i]);
        output_vec[i] = cast<Float2>(fp32x2_t{ix * norm_factor * (1.0f + wx), iy * norm_factor * (1.0f + wy)});
      }
      gmem.store(normed_ptr, output_vec, j);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType, bool kShared, bool kNorm>
struct HcFusedTailKernel {
  static_assert(sizeof(DType) == 2, "HcFusedTail only supports 2-byte dtypes");
  static_assert(kHiddenSize % 512 == 0, "kHiddenSize must be a multiple of 512 (grouped rmsnorm layout)");
  static constexpr auto kernel = hc_fused_tail_kernel<kHcCount, kHiddenSize, kUsePDL, DType, kShared, kNorm>;
  static constexpr auto kBlockSize = static_cast<uint32_t>(kHiddenSize / 16);

  static void
  run(const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView s,
      const tvm::ffi::TensorView h,
      const tvm::ffi::TensorView w_sg,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView norm_w,
      const tvm::ffi::TensorView out_hyper,
      const tvm::ffi::TensorView out_normed,
      float eps) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(y);
    TensorMatcher({M, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(out_hyper);
    TensorMatcher({M, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(logits);
    if constexpr (kShared) {
      TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(s).verify(h);
      TensorMatcher({kHiddenSize}).with_dtype<DType>().with_device(device).verify(w_sg);
    }
    if constexpr (kNorm) {
      TensorMatcher({M, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(out_normed);
      TensorMatcher({kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(norm_w);
    }

    const auto params = HcFusedTailParams{
        .y = y.data_ptr(),
        .s = s.data_ptr(),
        .h = h.data_ptr(),
        .w_sg = w_sg.data_ptr(),
        .residual = residual.data_ptr(),
        .logits = static_cast<const float*>(logits.data_ptr()),
        .norm_w = norm_w.data_ptr(),
        .out_hyper = out_hyper.data_ptr(),
        .out_normed = out_normed.data_ptr(),
        .eps = eps,
    };
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    if (num_tokens == 0) return;
    LaunchKernel(num_tokens * static_cast<uint32_t>(kHcCount), kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
