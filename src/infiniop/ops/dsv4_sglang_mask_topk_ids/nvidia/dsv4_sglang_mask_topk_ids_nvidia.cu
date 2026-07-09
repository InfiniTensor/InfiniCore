#include "../../../handle.h"
#include "dsv4_sglang_mask_topk_ids_nvidia.cuh"

#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>

namespace {
using mask_topk_ids_fn_t = void (*)(
    tvm::ffi::TensorView,
    tvm::ffi::TensorView);

constexpr const char *kMaskTopkIdsSymbol = "_Z20sglang_mask_topk_idsN3tvm3ffi10TensorViewES1_";

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

mask_topk_ids_fn_t resolveMaskTopkIds() {
    static mask_topk_ids_fn_t fn = []() -> mask_topk_ids_fn_t {
        tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
        tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL);
        void *handle = tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            return nullptr;
        }
        return reinterpret_cast<mask_topk_ids_fn_t>(dlsym(handle, kMaskTopkIdsSymbol));
    }();
    return fn;
}

tvm::ffi::TensorView makeTensorView(void *data,
                                    const std::array<int64_t, 2> &shape,
                                    const std::array<int64_t, 2> &strides,
                                    int device_id,
                                    DLTensor &tensor) {
    tensor.data = data;
    tensor.device = DLDevice{kDLROCM, device_id};
    tensor.ndim = 2;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = const_cast<int64_t *>(shape.data());
    tensor.strides = const_cast<int64_t *>(strides.data());
    tensor.byte_offset = 0;
    return tvm::ffi::TensorView(&tensor);
}

tvm::ffi::TensorView makeTensorView(const void *data,
                                    const std::array<int64_t, 1> &shape,
                                    const std::array<int64_t, 1> &strides,
                                    int device_id,
                                    DLTensor &tensor) {
    tensor.data = const_cast<void *>(data);
    tensor.device = DLDevice{kDLROCM, device_id};
    tensor.ndim = 1;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = const_cast<int64_t *>(shape.data());
    tensor.strides = const_cast<int64_t *>(strides.data());
    tensor.byte_offset = 0;
    return tvm::ffi::TensorView(&tensor);
}

} // namespace
#endif

namespace op::dsv4_sglang_mask_topk_ids::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t topk_ids_desc, infiniopTensorDescriptor_t num_token_non_padded_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, topk_ids_desc, num_token_non_padded_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *topk_ids, const void *num_token_non_padded, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveMaskTopkIds();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);

    const auto &i = _info;
    int device_id = this->device_id;
    std::array<int64_t, 2> ids_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.topk)};
    std::array<int64_t, 2> ids_strides{static_cast<int64_t>(i.topk), 1};
    std::array<int64_t, 1> count_shape{1};
    std::array<int64_t, 1> count_strides{1};

    DLTensor ids_t, count_t;
    auto ids_v = makeTensorView(topk_ids, ids_shape, ids_strides, device_id, ids_t);
    auto count_v = makeTensorView(num_token_non_padded, count_shape, count_strides, device_id, count_t);

    fn(ids_v, count_v);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}

} // namespace op::dsv4_sglang_mask_topk_ids::nvidia
