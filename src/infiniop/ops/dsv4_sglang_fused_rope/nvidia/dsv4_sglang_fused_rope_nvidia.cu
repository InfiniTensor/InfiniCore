#include "../../../handle.h"
#include "dsv4_sglang_fused_rope_nvidia.cuh"

#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <optional>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

namespace {
using sglang_fused_rope_fn_t = void (*)(
    tvm::ffi::TensorView,
    tvm::ffi::Optional<tvm::ffi::TensorView>,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    bool);

constexpr const char *kSglangFusedRopeSymbol = "_Z17sglang_fused_ropeN3tvm3ffi10TensorViewENS0_8OptionalIS1_vEES1_S1_b";

const char *deepseekOpsPath() {
    const char *env = std::getenv("DEEPSEEK_V4_OPS_SO");
    if (env && env[0] != '\0') {
        return env;
    }
    return "libdeepseek_v4_ops.so";
}

void *tryDlopen(const char *path, int flags) {
    dlerror();
    return dlopen(path, flags);
}

sglang_fused_rope_fn_t resolveSglangFusedRope() {
    static sglang_fused_rope_fn_t fn = []() -> sglang_fused_rope_fn_t {
        tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
        tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL);
        void *handle = tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            return nullptr;
        }
        return reinterpret_cast<sglang_fused_rope_fn_t>(dlsym(handle, kSglangFusedRopeSymbol));
    }();
    return fn;
}

tvm::ffi::TensorView makeTensorView(void *data,
                                    DLDataType dtype,
                                    const std::array<int64_t, 3> &shape,
                                    const std::array<int64_t, 3> &strides,
                                    int device_id,
                                    DLTensor &tensor) {
    tensor.data = data;
    tensor.device = DLDevice{kDLROCM, device_id};
    tensor.ndim = 3;
    tensor.dtype = dtype;
    tensor.shape = const_cast<int64_t *>(shape.data());
    tensor.strides = const_cast<int64_t *>(strides.data());
    tensor.byte_offset = 0;
    return tvm::ffi::TensorView(&tensor);
}

tvm::ffi::TensorView makeTensorView(const void *data,
                                    DLDataType dtype,
                                    const std::array<int64_t, 2> &shape,
                                    const std::array<int64_t, 2> &strides,
                                    int device_id,
                                    DLTensor &tensor) {
    tensor.data = const_cast<void *>(data);
    tensor.device = DLDevice{kDLROCM, device_id};
    tensor.ndim = 2;
    tensor.dtype = dtype;
    tensor.shape = const_cast<int64_t *>(shape.data());
    tensor.strides = const_cast<int64_t *>(strides.data());
    tensor.byte_offset = 0;
    return tvm::ffi::TensorView(&tensor);
}

tvm::ffi::TensorView makeTensorView(const void *data,
                                    DLDataType dtype,
                                    const std::array<int64_t, 1> &shape,
                                    const std::array<int64_t, 1> &strides,
                                    int device_id,
                                    DLTensor &tensor) {
    tensor.data = const_cast<void *>(data);
    tensor.device = DLDevice{kDLROCM, device_id};
    tensor.ndim = 1;
    tensor.dtype = dtype;
    tensor.shape = const_cast<int64_t *>(shape.data());
    tensor.strides = const_cast<int64_t *>(strides.data());
    tensor.byte_offset = 0;
    return tvm::ffi::TensorView(&tensor);
}

} // namespace
#endif

namespace op::dsv4_sglang_fused_rope::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t freqs_cis_desc, infiniopTensorDescriptor_t positions_desc, bool inverse) {
    Info info;
    CHECK_STATUS(createInfo(&info, q_desc, freqs_cis_desc, positions_desc, inverse));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *q, const void *freqs_cis, const void *positions, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveSglangFusedRope();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);

    const auto &i = _info;
    int device_id = this->device_id;
    std::array<int64_t, 3> q_shape{static_cast<int64_t>(i.tokens), static_cast<int64_t>(i.heads), 64};
    std::array<int64_t, 3> q_strides{static_cast<int64_t>(i.heads * 64), 64, 1};
    std::array<int64_t, 2> freqs_shape{static_cast<int64_t>(i.tokens), 64};
    std::array<int64_t, 2> freqs_strides{64, 1};
    std::array<int64_t, 1> pos_shape{static_cast<int64_t>(i.tokens)};
    std::array<int64_t, 1> pos_strides{1};

    DLTensor q_t, freqs_t, pos_t;
    auto q_v = makeTensorView(q, DLDataType{kDLBfloat, 16, 1}, q_shape, q_strides, device_id, q_t);
    auto freqs_v = makeTensorView(freqs_cis, DLDataType{kDLFloat, 32, 1}, freqs_shape, freqs_strides, device_id, freqs_t);
    auto pos_v = makeTensorView(positions, DLDataType{kDLInt, 64, 1}, pos_shape, pos_strides, device_id, pos_t);
    tvm::ffi::Optional<tvm::ffi::TensorView> k_v(std::nullopt);

    fn(q_v, k_v, freqs_v, pos_v, i.inverse);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}

} // namespace op::dsv4_sglang_fused_rope::nvidia
