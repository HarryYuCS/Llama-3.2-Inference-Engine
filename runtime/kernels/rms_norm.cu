/**

RMSNorm kernel specialized for hidden size 2048 (Llama-3.2-1B).

Launch contract:
  grid  = (seq_len,)   // one block per token / row
  block = (256,)       // or any power-of-two <= 1024 that divides 2048 evenly enough to cover via stride

x, y : [seq_len, 2048] BF16 row-major
gamma: [2048] BF16

*/

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace kernels {

namespace {

constexpr float kEpsilon = 1e-5f;
constexpr std::size_t kEmbeddingDim = 2048;

// Full-warp inclusive participation mask.
constexpr unsigned kFullWarpMask = 0xffffffffu;

__device__ float warpReduceSum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

}  // namespace

/**
RMS = sqrt(mean(x^2) + eps)
y_i = x_i / RMS * gamma_i
*/
__global__ void rmsNorm(
    const void* x,
    const void* gamma,
    void* y,
    std::size_t seqLen
) {
    const auto* xBf16 = static_cast<const nv_bfloat16*>(x);
    const auto* gammaBf16 = static_cast<const nv_bfloat16*>(gamma);
    auto* yBf16 = static_cast<nv_bfloat16*>(y);

    const std::size_t row = blockIdx.x;
    if (row >= seqLen) {
        return;
    }

    const std::size_t rowOffset = row * kEmbeddingDim;
    const unsigned tid = threadIdx.x;
    const unsigned blockSize = blockDim.x;

    // Partial sum of squares in FP32 (grid-stride over hidden dim).
    float partialSumSq = 0.0f;
    for (std::size_t h = tid; h < kEmbeddingDim; h += blockSize) {
        const float v = __bfloat162float(xBf16[rowOffset + h]);
        partialSumSq += v * v;
    }

    // Warp reduce, then block reduce via shared warp sums.
    partialSumSq = warpReduceSum(partialSumSq);

    // Max 32 warps per block (1024 threads).
    __shared__ float warpSums[32];
    __shared__ float invRms;

    const unsigned lane = tid & 31u;
    const unsigned warpId = tid >> 5u;
    if (lane == 0u) {
        warpSums[warpId] = partialSumSq;
    }
    __syncthreads();

    // smart cursor warpReduceSum 
    if (warpId == 0u) {
        const unsigned numWarps = (blockSize + 31u) / 32u;
        float blockSum = (lane < numWarps) ? warpSums[lane] : 0.0f;
        blockSum = warpReduceSum(blockSum);
        if (lane == 0u) {
            invRms = rsqrtf(blockSum / static_cast<float>(kEmbeddingDim) + kEpsilon);
        }
    }
    __syncthreads();

    const float scale = invRms;
    for (std::size_t h = tid; h < kEmbeddingDim; h += blockSize) {
        const float xv = __bfloat162float(xBf16[rowOffset + h]);
        const float gv = __bfloat162float(gammaBf16[h]);
        yBf16[rowOffset + h] = __float2bfloat16(xv * scale * gv);
    }
}

}  // namespace kernels
