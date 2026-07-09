#include "../../../handle.h"
#include "dsv4_deepgemm_tf32_hc_pernorm_gemm_nvidia.cuh"

#if defined(ENABLE_HYGON_API)
#include <cuda_bf16.h>

namespace {
__global__ void zeroFloatKernel(float *ptr, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ptr[idx] = 0.0f;
    }
}

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 32;

__global__ void tf32HcPernormTiledGemmKernel(const __nv_bfloat16 *a,
                                             const float *b,
                                             float *d,
                                             size_t m,
                                             size_t n,
                                             size_t k) {
    __shared__ float a_tile[kTileM][kTileK];
    __shared__ float b_tile[kTileN][kTileK];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTileM + ty;
    const int col = blockIdx.x * kTileN + tx;
    const int tid = ty * blockDim.x + tx;

    float acc = 0.0f;
    for (size_t k_base = 0; k_base < k; k_base += kTileK) {
        for (int idx = tid; idx < kTileM * kTileK; idx += kTileM * kTileN) {
            const int tile_row = idx / kTileK;
            const int tile_k = idx % kTileK;
            const size_t global_row = blockIdx.y * kTileM + tile_row;
            const size_t global_k = k_base + tile_k;
            a_tile[tile_row][tile_k] = (global_row < m && global_k < k)
                                         ? __bfloat162float(a[global_row * k + global_k])
                                         : 0.0f;
        }
        for (int idx = tid; idx < kTileN * kTileK; idx += kTileM * kTileN) {
            const int tile_col = idx / kTileK;
            const int tile_k = idx % kTileK;
            const size_t global_col = blockIdx.x * kTileN + tile_col;
            const size_t global_k = k_base + tile_k;
            b_tile[tile_col][tile_k] = (global_col < n && global_k < k)
                                         ? b[global_col * k + global_k]
                                         : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            acc += a_tile[ty][kk] * b_tile[tx][kk];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        d[row * n + col] = acc;
    }
}

__global__ void tf32HcPernormSqrSumKernel(const __nv_bfloat16 *a,
                                          float *sqr_sum,
                                          size_t m,
                                          size_t k) {
    const size_t row = blockIdx.x;
    if (row >= m) {
        return;
    }

    __shared__ float partial[256];
    float local = 0.0f;
    for (size_t kk = threadIdx.x; kk < k; kk += blockDim.x) {
        float av = __bfloat162float(a[row * k + kk]);
        local += av * av;
    }
    partial[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sqr_sum[row] = partial[0];
    }
}

infiniStatus_t launchTf32HcPernormGfx936(const op::dsv4_deepgemm_tf32_hc_pernorm_gemm::Info &info,
                                         const void *a,
                                         const void *b,
                                         void *d,
                                         void *sqr_sum,
                                         void *stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int block = 256;
    size_t d_count = info.d_ndim == 3 ? info.num_splits * info.m * info.n : info.m * info.n;
    size_t s_count = info.sqr_sum_ndim == 2 ? info.num_splits * info.m : info.m;
    zeroFloatKernel<<<(d_count + block - 1) / block, block, 0, cuda_stream>>>(reinterpret_cast<float *>(d), d_count);
    zeroFloatKernel<<<(s_count + block - 1) / block, block, 0, cuda_stream>>>(reinterpret_cast<float *>(sqr_sum), s_count);
    dim3 threads(kTileN, kTileM);
    dim3 grid((info.n + kTileN - 1) / kTileN, (info.m + kTileM - 1) / kTileM);
    tf32HcPernormTiledGemmKernel<<<grid, threads, 0, cuda_stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(a),
        reinterpret_cast<const float *>(b),
        reinterpret_cast<float *>(d),
        info.m,
        info.n,
        info.k);
    tf32HcPernormSqrSumKernel<<<info.m, block, 0, cuda_stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(a),
        reinterpret_cast<float *>(sqr_sum),
        info.m,
        info.k);
    return INFINI_STATUS_SUCCESS;
}

} // namespace
#endif

namespace op::dsv4_deepgemm_tf32_hc_pernorm_gemm::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t a_desc,
                                  infiniopTensorDescriptor_t b_desc,
                                  infiniopTensorDescriptor_t d_desc,
                                  infiniopTensorDescriptor_t sqr_sum_desc,
                                  int64_t num_splits) {
    Info info;
    CHECK_STATUS(createInfo(&info, a_desc, b_desc, d_desc, sqr_sum_desc, num_splits));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, const void *a, const void *b, void *d, void *sqr_sum, void *stream) const {
#if defined(ENABLE_HYGON_API)
    return launchTf32HcPernormGfx936(_info, a, b, d, sqr_sum, stream);
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}

} // namespace op::dsv4_deepgemm_tf32_hc_pernorm_gemm::nvidia
