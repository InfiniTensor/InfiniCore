#include "fp8_blockwise_gemm_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace op::fp8_blockwise_gemm::nvidia {
namespace {

// ---------------------------------------------------------------------------
// dtype helpers
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ float from_float<float>(float value) {
    return value;
}

// Load 4 consecutive elements (8B for half/bf16, 16B for float) and convert to
// float. The address is always 4-element aligned (K % 4 == 0 and lanes walk in
// steps of 4).
template <typename T>
__device__ __forceinline__ void load4(const T *p, float *v);

template <>
__device__ __forceinline__ void load4<half>(const half *p, float *v) {
    const half2 h01 = *reinterpret_cast<const half2 *>(p);
    const half2 h23 = *reinterpret_cast<const half2 *>(p + 2);
    const float2 f01 = __half22float2(h01);
    const float2 f23 = __half22float2(h23);
    v[0] = f01.x;
    v[1] = f01.y;
    v[2] = f23.x;
    v[3] = f23.y;
}

template <>
__device__ __forceinline__ void load4<__nv_bfloat16>(const __nv_bfloat16 *p, float *v) {
    const __nv_bfloat162 b01 = *reinterpret_cast<const __nv_bfloat162 *>(p);
    const __nv_bfloat162 b23 = *reinterpret_cast<const __nv_bfloat162 *>(p + 2);
    const float2 f01 = __bfloat1622float2(b01);
    const float2 f23 = __bfloat1622float2(b23);
    v[0] = f01.x;
    v[1] = f01.y;
    v[2] = f23.x;
    v[3] = f23.y;
}

template <>
__device__ __forceinline__ void load4<float>(const float *p, float *v) {
    const float4 f = *reinterpret_cast<const float4 *>(p);
    v[0] = f.x;
    v[1] = f.y;
    v[2] = f.z;
    v[3] = f.w;
}

// ---------------------------------------------------------------------------
// Fused FP8 blockwise GEMM: out[m, n] = sum_k a[m,k] * w[n,k] * s[n/BN, k/BK]
//
// Decode-oriented GEMV shape: one warp computes the M_TILE outputs of one
// weight row (warp-per-row, 4 warps per block). The 32 lanes of a warp cover
// a 512-byte K group (16 FP8 each), so each FP8 weight byte is read exactly
// once and reused for all M_TILE activation rows in registers; the next group
// is prefetched into registers while the current one is consumed. Per-128-
// chunk partial dots are scaled and accumulated in FP32. K tails that are not
// a multiple of 512 bytes fall back to a 4-byte path.
// ---------------------------------------------------------------------------

constexpr int TN = 4;           // weight rows per thread block (4 warps x 1 row)

template <typename T>
__device__ __forceinline__ void load16(const T *p, float *v);

template <>
__device__ __forceinline__ void load16<half>(const half *p, float *v) {
    load4(p, v);
    load4(p + 4, v + 4);
    load4(p + 8, v + 8);
    load4(p + 12, v + 12);
}

template <>
__device__ __forceinline__ void load16<__nv_bfloat16>(const __nv_bfloat16 *p, float *v) {
    load4(p, v);
    load4(p + 4, v + 4);
    load4(p + 8, v + 8);
    load4(p + 12, v + 12);
}

template <>
__device__ __forceinline__ void load16<float>(const float *p, float *v) {
    load4(p, v);
    load4(p + 4, v + 4);
    load4(p + 8, v + 8);
    load4(p + 12, v + 12);
}

__device__ __forceinline__ void decode16(uint4 q16, float *w) {
    w[0] = infiniopFp8E4m3Decode(q16.x & 0xffU);
    w[1] = infiniopFp8E4m3Decode((q16.x >> 8) & 0xffU);
    w[2] = infiniopFp8E4m3Decode((q16.x >> 16) & 0xffU);
    w[3] = infiniopFp8E4m3Decode(q16.x >> 24);
    w[4] = infiniopFp8E4m3Decode(q16.y & 0xffU);
    w[5] = infiniopFp8E4m3Decode((q16.y >> 8) & 0xffU);
    w[6] = infiniopFp8E4m3Decode((q16.y >> 16) & 0xffU);
    w[7] = infiniopFp8E4m3Decode(q16.y >> 24);
    w[8] = infiniopFp8E4m3Decode(q16.z & 0xffU);
    w[9] = infiniopFp8E4m3Decode((q16.z >> 8) & 0xffU);
    w[10] = infiniopFp8E4m3Decode((q16.z >> 16) & 0xffU);
    w[11] = infiniopFp8E4m3Decode(q16.z >> 24);
    w[12] = infiniopFp8E4m3Decode(q16.w & 0xffU);
    w[13] = infiniopFp8E4m3Decode((q16.w >> 8) & 0xffU);
    w[14] = infiniopFp8E4m3Decode((q16.w >> 16) & 0xffU);
    w[15] = infiniopFp8E4m3Decode(q16.w >> 24);
}

template <typename T, int M_TILE>
INFINIOP_CUDA_KERNEL fp8_blockwise_gemm_kernel(
    T *__restrict__ out,
    const T *__restrict__ a,
    const uint8_t *__restrict__ q,
    const float *__restrict__ scales,
    size_t M, size_t N, size_t K,
    size_t block_n, size_t block_k, size_t scales_cols) {
    const size_t row = static_cast<size_t>(blockIdx.x) * TN + (threadIdx.x >> 5);
    if (row >= N) {
        return;
    }
    const size_t m0 = static_cast<size_t>(blockIdx.y) * M_TILE;
    const int lane = threadIdx.x & 31;

    {
        const uint8_t *q_row = q + row * K;
        const size_t scale_row = row / block_n;

        float acc[M_TILE];
#pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            acc[m] = 0.0f;
        }

        // Main loop: 512-byte groups (lane*16 within group; lanes 0-7 cover the
        // first 128-wide sub-chunk, 8-15 the second, etc.).
        const size_t k_groups = K / 512;
        const size_t lane_off = static_cast<size_t>(lane) * 16;
        const size_t sub_chunk = static_cast<size_t>(lane) / 8;

        uint4 q_next = (k_groups > 0)
                         ? *reinterpret_cast<const uint4 *>(q_row + lane_off)
                         : make_uint4(0, 0, 0, 0);
        for (size_t kg = 0; kg < k_groups; ++kg) {
            const uint4 q_cur = q_next;
            const size_t k = (kg + 1) * 512 + lane_off;
            if (kg + 1 < k_groups) {
                q_next = *reinterpret_cast<const uint4 *>(q_row + k);
            }

            float w[16];
            decode16(q_cur, w);

            const size_t k_base = kg * 512 + sub_chunk * 128;
            const float scale = scales[scale_row * scales_cols + k_base / block_k];

            float cacc[M_TILE];
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                cacc[m] = 0.0f;
            }
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (m0 + m >= M) {
                    break;
                }
                float av[16];
                load16(a + (m0 + m) * K + kg * 512 + lane_off, av);
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    cacc[m] = fmaf(w[j], av[j], cacc[m]);
                }
            }
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                acc[m] = fmaf(scale, cacc[m], acc[m]);
            }
        }

        // Tail: remaining 128-wide chunks (K % 512 != 0), 4 bytes per lane.
        for (size_t kc = k_groups * 4; kc < K / 128; ++kc) {
            const size_t k = kc * 128 + static_cast<size_t>(lane) * 4;
            const uint32_t q4 = *reinterpret_cast<const uint32_t *>(q_row + k);
            float w[4];
            w[0] = infiniopFp8E4m3Decode(q4 & 0xffU);
            w[1] = infiniopFp8E4m3Decode((q4 >> 8) & 0xffU);
            w[2] = infiniopFp8E4m3Decode((q4 >> 16) & 0xffU);
            w[3] = infiniopFp8E4m3Decode(q4 >> 24);

            const float scale = scales[scale_row * scales_cols + (kc * 128) / block_k];
            float cacc[M_TILE];
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                cacc[m] = 0.0f;
            }
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (m0 + m >= M) {
                    break;
                }
                float av[4];
                load4(a + (m0 + m) * K + k, av);
                cacc[m] = w[0] * av[0] + w[1] * av[1] + w[2] * av[2] + w[3] * av[3];
            }
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                acc[m] = fmaf(scale, cacc[m], acc[m]);
            }
        }

        // Warp reduce and store.
#pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (m0 + m >= M) {
                break;
            }
            float v = acc[m];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, offset);
            }
            if (lane == 0) {
                out[(m0 + m) * N + row] = from_float<T>(v);
            }
        }
    }
}

template <typename T, int M_TILE>
void launch_mtile(T *out, const T *a, const uint8_t *q, const float *scales,
                  const Fp8BlockwiseGemmInfo &info, cudaStream_t stream) {
    dim3 grid((info.N + TN - 1) / TN, (info.M + M_TILE - 1) / M_TILE);
    fp8_blockwise_gemm_kernel<T, M_TILE><<<grid, 128, 0, stream>>>(
        out, a, q, scales, info.M, info.N, info.K,
        info.block_n, info.block_k, info.scales_cols);
}

template <typename T>
void launch(T *out, const T *a, const uint8_t *q, const float *scales,
            const Fp8BlockwiseGemmInfo &info, cudaStream_t stream) {
    const size_t m = info.M;
    if (m <= 1) {
        launch_mtile<T, 1>(out, a, q, scales, info, stream);
    } else if (m <= 2) {
        launch_mtile<T, 2>(out, a, q, scales, info, stream);
    } else if (m <= 4) {
        launch_mtile<T, 4>(out, a, q, scales, info, stream);
    } else if (m <= 8) {
        launch_mtile<T, 8>(out, a, q, scales, info, stream);
    } else {
        launch_mtile<T, 16>(out, a, q, scales, info, stream);
    }
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t scales_desc) {
    auto info = Fp8BlockwiseGemmInfo::create(out_desc, a_desc, q_desc, scales_desc);
    CHECK_RESULT(info);
    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{nvidia_handle->internal()}, info.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *, size_t, void *out,
    const void *a, const void *q, const void *scales, void *stream) const {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto q_ptr = reinterpret_cast<const uint8_t *>(q);
    auto scales_ptr = reinterpret_cast<const float *>(scales);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        launch(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(a), q_ptr, scales_ptr, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        launch(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(a), q_ptr, scales_ptr, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        launch(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(a), q_ptr, scales_ptr, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fp8_blockwise_gemm::nvidia
