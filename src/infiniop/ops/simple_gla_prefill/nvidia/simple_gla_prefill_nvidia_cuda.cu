#include "simple_gla_prefill_nvidia_cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

#include "../../../devices/nvidia/nvidia_handle.cuh"

namespace op::simple_gla_prefill_cuda::nvidia {

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float f16_to_f32(__half x) { return __half2float(x); }
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ __half f32_to_f16(float x) { return __float2half_rn(x); }

template <typename T>
struct Convert;

template <>
struct Convert<__half> {
    __device__ static float to_f32(__half x) { return f16_to_f32(x); }
    __device__ static __half from_f32(float x) { return f32_to_f16(x); }
};

template <>
struct Convert<__nv_bfloat16> {
    __device__ static float to_f32(__nv_bfloat16 x) { return bf16_to_f32(x); }
    __device__ static __nv_bfloat16 from_f32(float x) { return f32_to_bf16(x); }
};

// Naive but fused prefill kernel:
// - One block per (b,h)
// - Shared S[D*D] in fp32
// - Loop over t and update S + compute o_t
template <typename T>
__global__ void simple_gla_prefill_kernel(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    const float *g_gamma,
    int B,
    int Tlen,
    int H,
    int D,
    float scale) {

    const int b = (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;

    extern __shared__ float smem[];
    float *S = smem;               // D*D
    float *kvec = S + D * D;       // D
    float *vvec = kvec + D;        // D
    float *qvec = vvec + D;        // D

    // Initialize S to 0.
    for (int idx = tid; idx < D * D; idx += (int)blockDim.x) {
        S[idx] = 0.0f;
    }
    __syncthreads();

    const float gate = expf(g_gamma[h]);

    // Base pointers (contiguous [B,T,H,D])
    const int stride_b = Tlen * H * D;
    const int stride_t = H * D;
    const int stride_h = D;

    for (int t = 0; t < Tlen; ++t) {
        const int base = b * stride_b + t * stride_t + h * stride_h;

        // Load q/k/v vectors to shared (fp32).
        if (tid < D) {
            qvec[tid] = Convert<T>::to_f32(q[base + tid]);
            kvec[tid] = Convert<T>::to_f32(k[base + tid]);
            vvec[tid] = Convert<T>::to_f32(v[base + tid]);
        }
        __syncthreads();

        // Update S = S*gate + outer(k, v)
        for (int idx = tid; idx < D * D; idx += (int)blockDim.x) {
            const int dk = idx / D;
            const int dv = idx - dk * D;
            S[idx] = S[idx] * gate + kvec[dk] * vvec[dv];
        }
        __syncthreads();

        // Compute out[t, d] for this (b,h) using D threads.
        if (tid < D) {
            float acc = 0.0f;
            const int dv = tid;
            for (int dk = 0; dk < D; ++dk) {
                acc += (qvec[dk] * scale) * S[dk * D + dv];
            }
            out[base + dv] = Convert<T>::from_f32(acc);
        }
        __syncthreads();
    }
}

// Chunked/tiled kernel for D > 64 (e.g. D=128) to stay under 64KB shared memory.
// Grid: (ceil(D/BK), ceil(D/BV), B*H), block: 256 threads. Each block holds S_tile [BK][BV].
constexpr int TILE = 32;

template <typename T>
__global__ void simple_gla_prefill_chunked_kernel(
    float *out_float,
    const T *q,
    const T *k,
    const T *v,
    const float *g_gamma,
    int B,
    int Tlen,
    int H,
    int D,
    float scale) {

    const int i_k = (int)blockIdx.x;  // tile index along K
    const int i_v = (int)blockIdx.y;  // tile index along V
    const int bh = (int)blockIdx.z;
    const int b = bh / H;
    const int h = bh % H;
    const int tid = (int)threadIdx.x;

    const int k0 = i_k * TILE;
    const int v0 = i_v * TILE;
    const int nk = (D - k0) < TILE ? (D - k0) : TILE;
    const int nv = (D - v0) < TILE ? (D - v0) : TILE;

    extern __shared__ float smem[];
    float *S_tile = smem;  // [TILE][TILE] = 32*32*4 = 4KB
    float *q_tile = S_tile + TILE * TILE;
    float *k_tile = q_tile + TILE;
    float *v_tile = k_tile + TILE;

    const float gate = expf(g_gamma[h]);
    const int stride_b = Tlen * H * D;
    const int stride_t = H * D;
    const int stride_h = D;
    const int out_stride_b = Tlen * H * D;
    const int out_stride_t = H * D;
    const int out_stride_h = D;

    // Initialize S_tile to 0
    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        S_tile[i] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < Tlen; ++t) {
        const int base = b * stride_b + t * stride_t + h * stride_h;

        // Load q, k, v tiles into shared (fp32)
        if (tid < nk) {
            q_tile[tid] = Convert<T>::to_f32(q[base + k0 + tid]);
            k_tile[tid] = Convert<T>::to_f32(k[base + k0 + tid]);
        }
        if (tid < nv) {
            v_tile[tid] = Convert<T>::to_f32(v[base + v0 + tid]);
        }
        __syncthreads();

        // S_tile = S_tile * gate + outer(k_tile, v_tile)
        for (int idx = tid; idx < TILE * TILE; idx += blockDim.x) {
            const int dk = idx / TILE;
            const int dv = idx - dk * TILE;
            if (dk < nk && dv < nv) {
                S_tile[idx] = S_tile[idx] * gate + k_tile[dk] * v_tile[dv];
            }
        }
        __syncthreads();

        // Output uses the updated state (matches naive kernel: update S then compute o_t).
        if (tid < nv) {
            float acc = 0.0f;
            for (int kk = 0; kk < nk; ++kk) {
                acc += (q_tile[kk] * scale) * S_tile[kk * TILE + tid];
            }
            atomicAdd(&out_float[b * out_stride_b + t * out_stride_t + h * out_stride_h + v0 + tid], acc);
        }
        __syncthreads();
    }
}

// Convert float buffer (B,T,H,D) to output dtype.
template <typename T>
__global__ void simple_gla_prefill_convert_kernel(
    T *out,
    const float *in_float,
    int B,
    int Tlen,
    int H,
    int D) {
    const int idx = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * Tlen * H * D;
    if (idx < total) {
        out[idx] = Convert<T>::from_f32(in_float[idx]);
    }
}

} // namespace

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_gamma_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto info_r = op::simple_gla_prefill_cuda::SimpleGLAPrefillCudaInfo::create(
        out_desc, q_desc, k_desc, v_desc, g_gamma_desc);
    if (!info_r) return info_r.status();
    auto info = info_r.take();

    // Workspace for chunked path (D > 64): float buffer B*T*H*D.
    const size_t D_val = info.D;
    const size_t workspace_size = (D_val > 64u)
        ? (info.B * info.T * info.H * info.D * sizeof(float))
        : 0u;
    *desc_ptr = new Descriptor(
        /*opaque=*/nullptr,
        info,
        workspace_size,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *g_gamma,
    float scale,
    void *stream) const {

    const int B = (int)_info.B;
    const int Tlen = (int)_info.T;
    const int H = (int)_info.H;
    const int D = (int)_info.D;
    const bool use_chunked = (D > 64);
    cudaStream_t cuda_stream = (cudaStream_t)stream;

    if (use_chunked) {
        // Chunked path: write to workspace (float), then convert to out.
        const int nk_tiles = (D + TILE - 1) / TILE;
        const int nv_tiles = (D + TILE - 1) / TILE;
        dim3 grid_chunk(nk_tiles, nv_tiles, B * H);
        dim3 block_chunk(256, 1, 1);
        const size_t shmem_chunk = sizeof(float) * (TILE * TILE + 3 * TILE);
        if (workspace == nullptr || workspace_size < (size_t)(B * Tlen * H * D * (int)sizeof(float))) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        float *out_float = static_cast<float *>(workspace);
        cudaMemsetAsync(out_float, 0, (size_t)(B * Tlen * H * D) * sizeof(float), cuda_stream);
        if (_info.dtype == INFINI_DTYPE_F16) {
            simple_gla_prefill_chunked_kernel<<<grid_chunk, block_chunk, shmem_chunk, cuda_stream>>>(
                out_float, (const half *)q, (const half *)k, (const half *)v, (const float *)g_gamma,
                B, Tlen, H, D, scale);
            const int total = B * Tlen * H * D;
            simple_gla_prefill_convert_kernel<<<(total + 255) / 256, 256, 0, cuda_stream>>>(
                (half *)out, out_float, B, Tlen, H, D);
        } else if (_info.dtype == INFINI_DTYPE_BF16) {
            simple_gla_prefill_chunked_kernel<<<grid_chunk, block_chunk, shmem_chunk, cuda_stream>>>(
                out_float, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, (const float *)g_gamma,
                B, Tlen, H, D, scale);
            const int total = B * Tlen * H * D;
            simple_gla_prefill_convert_kernel<<<(total + 255) / 256, 256, 0, cuda_stream>>>(
                (__nv_bfloat16 *)out, out_float, B, Tlen, H, D);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        return INFINI_STATUS_SUCCESS;
    }

    // Naive path (D <= 64)
    (void)workspace;
    (void)workspace_size;
    dim3 grid(B, H, 1);
    dim3 block(256, 1, 1);
    const size_t shmem = sizeof(float) * (D * D + 3 * D);

    if (_info.dtype == INFINI_DTYPE_F16) {
        simple_gla_prefill_kernel<<<grid, block, shmem, cuda_stream>>>(
            (half *)out, (const half *)q, (const half *)k, (const half *)v, (const float *)g_gamma,
            B, Tlen, H, D, scale);
        return INFINI_STATUS_SUCCESS;
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        simple_gla_prefill_kernel<<<grid, block, shmem, cuda_stream>>>(
            (__nv_bfloat16 *)out, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, (const float *)g_gamma,
            B, Tlen, H, D, scale);
        return INFINI_STATUS_SUCCESS;
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::simple_gla_prefill_cuda::nvidia
