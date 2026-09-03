#include "block_fp8_linear_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#ifdef ENABLE_CUTLASS_API
#include "block_fp8_gemm_sm120.cuh"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cmath>

namespace op::block_fp8_linear::nvidia {

namespace {

__global__ void per_token_group_quant_kernel(
    const __nv_bfloat16 *__restrict__ input,
    __nv_fp8_e4m3 *__restrict__ output,
    float *__restrict__ scales,
    size_t M,
    size_t K,
    size_t num_groups) {
    const size_t row = blockIdx.x;
    if (row >= M) return;

    const size_t tid = threadIdx.x;
    const size_t threads_per_group = 8;
    const size_t group_id = tid / threads_per_group;
    const size_t lane = tid % threads_per_group;
    const size_t groups_per_block = blockDim.x / threads_per_group;

    for (size_t g = group_id; g < num_groups; g += groups_per_block) {
        size_t start = g * 128;
        size_t end = min(start + 128, K);

        float local_max = 0.0f;
        for (size_t i = lane; i < 128 && (start + i) < K; i += threads_per_group) {
            float val = static_cast<float>(input[row * K + start + i]);
            local_max = fmaxf(local_max, fabsf(val));
        }

        for (int offset = 4; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xff, local_max, offset));
        }

        float amax = local_max;
        float scale = fmaxf(amax / 448.0f, 1e-10f);
        if (lane == 0) {
            scales[row * num_groups + g] = scale;
        }

        float inv_scale = 1.0f / scale;
        for (size_t i = lane; i < 128 && (start + i) < K; i += threads_per_group) {
            float val = static_cast<float>(input[row * K + start + i]);
            float q = val * inv_scale;
            q = fminf(fmaxf(q, -448.0f), 448.0f);
            output[row * K + start + i] = __nv_fp8_e4m3(q);
        }
    }
}

void launch_per_token_group_quant(
    const __nv_bfloat16 *input,
    __nv_fp8_e4m3 *output,
    float *scales,
    size_t M,
    size_t K,
    size_t num_groups,
    cudaStream_t stream) {
    int threads = 128;
    int blocks = static_cast<int>(M);
    per_token_group_quant_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scales, M, K, num_groups);
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    size_t workspace_size;
};

Descriptor::~Descriptor() { delete _opaque; }

size_t Descriptor::workspaceSize() const {
    return _opaque->workspace_size;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc) {
    auto info_result = BlockFP8LinearInfo::create(
        output_desc, input_desc, weight_desc, weight_scale_desc);
    CHECK_RESULT(info_result);

    auto info = info_result.take();

    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);

    const size_t M = info.M;
    const size_t K = info.K;
    const size_t N = info.N;
    const size_t num_in_blocks = info.num_in_blocks;

    auto align256 = [](size_t n) { return (n + 255) & ~255; };

    size_t a_fp8_size = align256(M * K);
    size_t a_scale_size = align256(M * num_in_blocks * 4);

#ifdef ENABLE_CUTLASS_API
    size_t cutlass_ws = sm120::get_gemm_workspace_size(M, N, K);
#else
    size_t cutlass_ws = 0;
#endif
    cutlass_ws = align256(cutlass_ws);

    size_t total_ws = a_fp8_size + a_scale_size + cutlass_ws;

    *desc_ptr = new Descriptor(
        new Opaque{nvidia_handle->internal(), total_ws},
        info, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *weight_scale,
    void *stream) const {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    const size_t M = _info.M;
    const size_t N = _info.N;
    const size_t K = _info.K;
    const size_t num_in_blocks = _info.num_in_blocks;

    auto align256 = [](size_t n) { return (n + 255) & ~255; };

    size_t a_fp8_offset = 0;
    size_t a_fp8_size = align256(M * K);
    size_t a_scale_offset = a_fp8_offset + a_fp8_size;
    size_t a_scale_size = align256(M * num_in_blocks * 4);
    size_t cutlass_offset = a_scale_offset + a_scale_size;

    auto *a_fp8 = static_cast<__nv_fp8_e4m3 *>(static_cast<void *>(
        static_cast<char *>(workspace) + a_fp8_offset));
    auto *a_scales = static_cast<float *>(static_cast<void *>(
        static_cast<char *>(workspace) + a_scale_offset));
    auto *cutlass_ws = static_cast<void *>(
        static_cast<char *>(workspace) + cutlass_offset);

    auto *a_bf16 = reinterpret_cast<const __nv_bfloat16 *>(input);
    auto *w_fp8 = reinterpret_cast<const __nv_fp8_e4m3 *>(weight);
    auto *w_scales = reinterpret_cast<const float *>(weight_scale);
    auto *out_bf16 = reinterpret_cast<__nv_bfloat16 *>(output);

    launch_per_token_group_quant(
        a_bf16, a_fp8, a_scales, M, K, num_in_blocks, cuda_stream);

#ifdef ENABLE_CUTLASS_API
    auto status = sm120::run_gemm(
        out_bf16, a_fp8, a_scales, w_fp8, w_scales,
        M, N, K, cutlass_ws, cuda_stream);

    if (status != cutlass::Status::kSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::block_fp8_linear::nvidia
