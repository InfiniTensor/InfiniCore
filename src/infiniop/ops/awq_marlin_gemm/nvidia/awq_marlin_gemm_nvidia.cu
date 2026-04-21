#if defined(ENABLE_NVIDIA_API)
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../core/utils.h"
#include "awq_marlin_gemm_nvidia.cuh"
#include "kernel.cuh"

template <typename scalar_t, typename Tdata>
infiniStatus_t awq_marlin_gemm_kernel(
    const void *a,
    void *c,
    const void *b_q_weight,
    void *b_bias,
    void *b_scales,
    void *a_scales,
    void *global_scale,
    void *b_zeros,
    void *g_idx,
    void *perm,
    int64_t b_type_id,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    int size_m,
    int size_k,
    int size_n,
    int b_q_size_0,
    int b_q_size_1,
    int a_stride_0,
    int b_zeros_size_1,
    int num_groups,
    void *total_buffer,
    cudaStream_t stream) {
    // scalar_t *a, Tdata *b_scales
    vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

    if constexpr (std::is_same<scalar_t, half>::value) {
        a_type_id = vllm::kFloat16.id();
        c_type_id = vllm::kFloat16.id();
    } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
        a_type_id = vllm::kBFloat16.id();
        c_type_id = vllm::kBFloat16.id();
    } else {
        // 此时c和b_scales类型相同
        if constexpr (std::is_same<Tdata, half>::value) {
            c_type_id = vllm::kFloat16.id();
        } else if constexpr (std::is_same<Tdata, nv_bfloat16>::value) {
            c_type_id = vllm::kBFloat16.id();
        } else {
            c_type_id = vllm::kBFloat16.id();
            host::RuntimeCheck(c != nullptr, "c must be passed for W4A8-FP4\n");
        }
        if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
            a_type_id = vllm::kFE4M3fn.id();
        } else if constexpr (std::is_same<scalar_t, char>::value) {
            a_type_id = vllm::kS8.id();
        } else {
            host::RuntimeCheck(false, "unsupported `a` scalar_type\n");
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    s_type_id = c_type_id;
    if (b_type_id == vllm::kFE2M1f.id()) {
        if constexpr (std::is_same<Tdata, __nv_fp8_e4m3>::value) {
            s_type_id = vllm::kFE4M3fn.id();
        } else if constexpr (std::is_same<Tdata, uint8_t>::value) {
            printf("b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu\n");
            s_type_id = vllm::kFE8M0fnu.id();
        } else {
            host::RuntimeCheck(false,
                               "When b_type = float4_e2m1f, b_scale scalar type must be",
                               "float8_e4m3fn (for NVFP4) or float8_e8m0fnu (for MXFP4).");
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
    vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
    vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
    vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

    int pack_factor = 32 / b_type.size_bits();

    // Verify a = [size_m, size_k]

    // Verify b
    host::RuntimeCheck(
        size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
        " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
    host::RuntimeCheck((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_size_0,
                       "Shape mismatch: b_q_weight.size(0) = ", b_q_size_0,
                       ", size_k = ", size_k,
                       ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
    host::RuntimeCheck(
        b_q_size_1 % MARLIN_NAMESPACE_NAME::tile_size == 0,
        "b_q_weight.size(1) = ", b_q_size_1,
        " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
    int actual_size_n = (b_q_size_1 / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
    host::RuntimeCheck(size_n == actual_size_n, "size_n = ", size_n,
                       ", actual_size_n = ", actual_size_n);

    // Verify device and strides

    // We use int4 (16 bytes) to load A, so A must aligned to 16 bytes
    host::RuntimeCheck(a_stride_0 % 8 == 0, "A.stride(0) must divisible by 8");
    host::RuntimeCheck(reinterpret_cast<uintptr_t>(a) % 16 == 0, "A must aligned to 16 bytes");

    if (a_scales != nullptr) {
        host::RuntimeCheck(a_type.size_bits() == 8,
                           "a_scales can only be used for 8bit activation.");
    } else {
        host::RuntimeCheck(a_type.size_bits() != 8,
                           "the a_scales parameter must be passed for 8bit activation.");
    }

    int device_id = 0;
    // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
    // auto -1)
    int thread_k = -1;
    // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
    // auto -1)
    int thread_n = -1;
    // sms: number of SMs to use for the kernel
    int sms = -1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);

    // Alloc buffers
    float *c_tmp = nullptr;
    void *a_tmp = nullptr;
    void *workspace = nullptr;

    int c_tmp_bytes = 0;
    // Alloc C tmp buffer that is going to be used for the global reduce

    if (use_fp32_reduce) {
        int max_m_block_size = (size_m + 16 - 1) / 16 * 16;
        max_m_block_size = min(max_m_block_size, 64);
        int max_c_tmp_size = sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
        c_tmp_bytes = max_c_tmp_size * sizeof(float);
    }

    // Detect groupsize and act_order

    // b_scales = [num_groups, size_n]
    // g_idx.size(-1) == size_k && perm.size(-1) == size_k
    int a_tmp_bytes = 0;
    bool has_act_order = false;

    if (g_idx != nullptr && perm != nullptr) {
        has_act_order = true;
    }
    int group_size = -1;
    if (has_act_order) {
        a_tmp_bytes = size_m * size_k * sizeof(scalar_t);
        if (is_k_full) {
            host::RuntimeCheck(num_groups > 1, "For act_order, num_groups must be > 1");
            host::RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k,
                               ", is not divisible by num_groups = ", num_groups);
            group_size = size_k / num_groups;
        } else {
            group_size = 0;
        }
    } else {
        if (num_groups > 1) {
            host::RuntimeCheck(
                size_k % num_groups == 0, "size_k = ", size_k,
                ", is not divisible by b_scales.size(0) = ", num_groups);
            group_size = size_k / num_groups;
        } else {
            group_size = -1;
        }
    }

    int workspace_bytes = sms * sizeof(int64_t);
    const int total_bytes = c_tmp_bytes + a_tmp_bytes + workspace_bytes;
    // ===================== 3. 单次 cudaMalloc 分配 =====================
    if (total_bytes > 0) {

        cudaMemset(total_buffer, 0, total_bytes);
    }
    // ===================== 4. 手动切分指针（核心！） =====================
    uint8_t *ptr = reinterpret_cast<uint8_t *>(total_buffer);
    // 分配 c_tmp
    if (use_fp32_reduce && c_tmp_bytes > 0) {
        c_tmp = reinterpret_cast<float *>(ptr);
        ptr += c_tmp_bytes;
    }
    // 分配 a_tmp
    if (has_act_order && a_tmp_bytes > 0) {
        a_tmp = ptr;
        ptr += a_tmp_bytes;
    }

    // 分配 workspace
    if (workspace_bytes > 0) {
        workspace = ptr;
        ptr += workspace_bytes;
    }

    if (global_scale != nullptr) {

        host::RuntimeCheck(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
                           "global_scale can only be used for nvfp4 format.");

    } else {
        host::RuntimeCheck(!(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
                           "the global_scale parameter must be passed for nvfp4 format.");
    }
    // b_bias = [size_n, 1]
    bool has_bias = (b_bias != nullptr);

    bool has_zp = (b_zeros != nullptr);
    if (has_zp) {
        host::RuntimeCheck(
            b_type == vllm::kU4 || b_type == vllm::kU8,
            "b_type must be u4 or u8 when has_zp = True. Got = ", b_type.str());

    } else {
        host::RuntimeCheck(b_type == vllm::kU4B8 || b_type == vllm::kU8B128 || b_type == vllm::kS4 || b_type == vllm::kS8 || b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
                           "b_type must be uint4b8, uint8b128, int4, int8, "
                           "float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
                           b_type.str());
    }

    if (has_zp && is_zp_float) {
        if constexpr (!std::is_same<scalar_t, half>::value) {
            printf("Computation a_type must be float16 (half) when using float zero "
                   "points.\n");
        }
    }

    // Verify b_zeros
    if (has_zp) {
        if (is_zp_float) {
            // b_zeros = [num_groups, size_n]
            host::RuntimeCheck(b_zeros_size_1 == size_n,
                               "b_zeros dim 1 = ", b_zeros_size_1,
                               " is not size_n = ", size_n);
            host::RuntimeCheck(num_groups != -1, "num_groups must be != -1");
        } else {

            host::RuntimeCheck(b_zeros_size_1 == size_n / pack_factor,
                               "b_zeros dim 1 = ", b_zeros_size_1,
                               " is not size_n / pack_factor = ", size_n / pack_factor);
        }
    }

    // Verify workspace size
    host::RuntimeCheck(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
                       "size_n = ", size_n, ", is not divisible by min_thread_n = ",
                       MARLIN_NAMESPACE_NAME::min_thread_n);

    // a_scales和global_scale都必须是float *

    if (a_type.size_bits() == 16) {
        host::RuntimeCheck((a_type == c_type), "scalar type of a must be the same with c for 16 bit activation\n");
    }

    marlin::marlin_mm(
        a, b_q_weight, c, c_tmp,
        b_bias, a_scales, b_scales,
        global_scale, b_zeros, g_idx,
        perm, a_tmp, size_m, size_n, size_k, a_stride_0,
        workspace, a_type, b_type, c_type, s_type, has_bias,
        has_act_order, is_k_full, has_zp, num_groups, group_size, device_id,
        stream, thread_k, thread_n, sms,
        use_atomic_add, use_fp32_reduce, is_zp_float);
    return INFINI_STATUS_SUCCESS;
}

namespace op::awq_marlin_gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_bias_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t a_scales_desc,
    infiniopTensorDescriptor_t global_scales_desc,
    infiniopTensorDescriptor_t b_zeros_desc,
    infiniopTensorDescriptor_t g_idx_desc,
    infiniopTensorDescriptor_t perm_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto result = AwqMarlinGemmInfo::create(out_desc, a_desc, b_desc, b_bias_desc, b_scales_desc, a_scales_desc, global_scales_desc, b_zeros_desc, g_idx_desc, perm_desc);
    size_t size_m = a_desc->dim(0);
    size_t size_k = a_desc->dim(1);
    int a_tmp_bytes = 0;
    int c_tmp_bytes = 0;

    int device_id = 0;
    int sms = -1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);
    int workspace_bytes = sms * sizeof(int64_t);
    int max_m_block_size = (size_m + 16 - 1) / 16 * 16;
    max_m_block_size = min(max_m_block_size, 64);
    int max_c_tmp_size = sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
    c_tmp_bytes = max_c_tmp_size * sizeof(float);

    a_tmp_bytes = size_m * size_k * infiniSizeOf(a_desc->dtype());

    size_t workspace_size = c_tmp_bytes + a_tmp_bytes + workspace_bytes;

    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        result.take(),
        workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t
Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *b_bias,
    void *b_scales,
    void *a_scales,
    void *global_scales,
    void *b_zeros,
    void *g_idx,
    void *perm,
    int64_t b_q_type_id,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;
    int size_m = static_cast<int>(_info.size_m);
    int size_k = static_cast<int>(_info.size_k);
    int size_n = static_cast<int>(_info.size_n);
    int b_q_size_0 = static_cast<int>(_info.b_q_size_0);
    int b_q_size_1 = static_cast<int>(_info.b_q_size_1);
    int b_zeros_size_1 = static_cast<int>(_info.b_zeros_size_1);
    int a_stride_0 = static_cast<int>(_info.a_stride_0);
    int num_groups = _info.num_groups;

#define MARLIN(SCALAR_T, TDATA) \
    awq_marlin_gemm_kernel<SCALAR_T, TDATA>(a, c, b, b_bias, b_scales, a_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float, size_m, size_k, size_n, b_q_size_0, b_q_size_1, a_stride_0, b_zeros_size_1, num_groups, workspace, stream)

    if (_info.a_dtype == INFINI_DTYPE_F16) {
        return MARLIN(half, half);
    } else if (_info.a_dtype == INFINI_DTYPE_BF16) {
        return MARLIN(__nv_bfloat16, __nv_bfloat16);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::awq_marlin_gemm::nvidia
#endif
