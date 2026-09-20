#if defined(ENABLE_NVIDIA_API)

#include "linear_gguf_nvidia.cuh"

// Opens namespace op::linear_gguf::nvidia itself, so it has to stay at file
// scope rather than move inside that namespace.
#include "linear_gguf_gemv.cuh"
#include "linear_gguf_dequant.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace op::linear_gguf::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

namespace {

bool force_decode_cublas() {
    const char *value = std::getenv("INFINI_GGUF_DECODE_CUBLAS");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

bool strict_small_prefill(int32_t ggml_type) {
    const char *value = std::getenv("INFINI_GGUF_STRICT_SMALL_PREFILL");
    if (value == nullptr || std::strcmp(value, "0") == 0) {
        return false;
    }
    // Optional diagnostic filter.  When present, only this GGML weight type
    // gets the short-prefill register path; e.g. 12=Q4_K, 13=Q5_K,
    // 14=Q6_K, 8=Q8_0.  Omission preserves the original all-types experiment.
    const char *type_value =
        std::getenv("INFINI_GGUF_STRICT_SMALL_PREFILL_TYPE");
    return type_value == nullptr || *type_value == '\0'
        || std::atoi(type_value) == ggml_type;
}

size_t strict_small_prefill_limit(int32_t ggml_type) {
    constexpr size_t kDefaultDecodeM = 8;
    if (!strict_small_prefill(ggml_type)) {
        return kDefaultDecodeM;
    }
    const char *max_m_value =
        std::getenv("INFINI_GGUF_STRICT_SMALL_PREFILL_MAX_M");
    if (max_m_value != nullptr && *max_m_value != '\0') {
        const int requested = std::atoi(max_m_value);
        if (requested > static_cast<int>(kDefaultDecodeM)
            && requested <= kMaxDecodeM) {
            return static_cast<size_t>(requested);
        }
    }
    return static_cast<size_t>(kMaxDecodeM);
}

bool use_register_gemv(size_t m_count, int32_t ggml_type) {
    const size_t limit = strict_small_prefill_limit(ggml_type);
    return !force_decode_cublas() && m_count <= limit;
}

bool quantize_decode_activation(int32_t ggml_type) {
    const char *type_value = std::getenv("INFINI_GGUF_DECODE_Q8A_TYPE");
    if (type_value != nullptr && *type_value != '\0') {
        return std::atoi(type_value) == ggml_type;
    }
    const char *value = std::getenv("INFINI_GGUF_DECODE_Q8A");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

}  // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t w_desc,
    int64_t ggml_type) {
    auto info_result = LinearGgufInfo::create(out_desc, a_desc, w_desc, ggml_type);
    if (!info_result) {
        return info_result.status();
    }
    auto info = info_result.take();

    // The register-resident GEMV only covers small batches.  A larger M goes to
    // the prefill path, which needs one BF16 weight tile as scratch, so ask for it
    // here -- the infinicore wrapper sizes its workspace tensor from
    // workspaceSize().  M <= kMaxDecodeM keeps a zero-size workspace, leaving the
    // decode path exactly as cheap as it was.
    size_t workspace_size = 0;
    if (!use_register_gemv(info.m_count, info.ggml_type)) {
        workspace_size = prefill_scratch_bytes(static_cast<int64_t>(info.k_count));
    }

    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{nvidia_handle->internal()}, std::move(info), workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

// All of the prefill composition (tile decode + gemm) lives in
// linear_gguf_dequant.cuh::launch_prefill, because scripts/gguf_routeb_gemv_probe.cu
// drives exactly that function through its own cublas handle.  Keeping it here
// instead would have the numerical gate test a copy of the shipped path, which is
// the mistake stage 3.2 had to undo (log A.8).
infiniStatus_t run_prefill(
    const std::shared_ptr<device::nvidia::Handle::Internal> &internal,
    const LinearGgufInfo &info,
    void *workspace, size_t workspace_size, void *out,
    const void *a, const void *w, cudaStream_t stream) {
    bool ok = false;
    CHECK_STATUS(internal->useCublas(stream, [&](cublasHandle_t blas) {
        ok = launch_prefill(
            blas, info.ggml_type,
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const uint8_t *>(w),
            reinterpret_cast<__nv_bfloat16 *>(out),
            static_cast<int>(info.m_count), static_cast<int>(info.n_count),
            static_cast<int>(info.k_count), info.row_bytes,
            workspace, workspace_size, stream);
        return INFINI_STATUS_SUCCESS;
    }));
    return ok ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

}  // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *a, const void *w,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (!use_register_gemv(_info.m_count, _info.ggml_type)) {
        if (_info.out_is_f32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        return run_prefill(_opaque->internal, _info, workspace, workspace_size, out, a, w,
                           reinterpret_cast<cudaStream_t>(stream));
    }

    // The kernel decodes weight blocks on the fly; nothing here touches a
    // dequantized copy of the weight.
    const bool ok = _info.out_is_f32
        ? launch_gemv_decode_f32(
            _info.ggml_type,
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const uint8_t *>(w),
            reinterpret_cast<float *>(out),
            static_cast<int>(_info.m_count),
            static_cast<int>(_info.n_count),
            static_cast<int>(_info.k_count),
            _info.row_bytes,
            quantize_decode_activation(_info.ggml_type),
            reinterpret_cast<cudaStream_t>(stream))
        : launch_gemv_decode(
            _info.ggml_type,
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const uint8_t *>(w),
            reinterpret_cast<__nv_bfloat16 *>(out),
            static_cast<int>(_info.m_count),
            static_cast<int>(_info.n_count),
            static_cast<int>(_info.k_count),
            _info.row_bytes,
            quantize_decode_activation(_info.ggml_type),
            reinterpret_cast<cudaStream_t>(stream));
    if (!ok) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::linear_gguf::nvidia

#endif  // ENABLE_NVIDIA_API
