#include "../../../handle.h"
#include "dsv4_sglang_hash_topk_nvidia.cuh"

#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>

namespace {
using hash_topk_fn_t = void (*)(
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    double);

constexpr const char *kHashTopkSymbol = "_Z29sglang_hash_topk_sqrtsoftplusN3tvm3ffi10TensorViewES1_S1_S1_S1_d";

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

hash_topk_fn_t resolveHashTopk() {
    static hash_topk_fn_t fn = []() -> hash_topk_fn_t {
        tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
        tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL);
        void *handle = tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            return nullptr;
        }
        return reinterpret_cast<hash_topk_fn_t>(dlsym(handle, kHashTopkSymbol));
    }();
    return fn;
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

tvm::ffi::TensorView makeTensorView(void *data,
                                    DLDataType dtype,
                                    const std::array<int64_t, 2> &shape,
                                    const std::array<int64_t, 2> &strides,
                                    int device_id,
                                    DLTensor &tensor) {
    tensor.data = data;
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

namespace op::dsv4_sglang_hash_topk::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t router_logits_desc,
                                  infiniopTensorDescriptor_t input_ids_desc,
                                  infiniopTensorDescriptor_t tid2eid_desc,
                                  infiniopTensorDescriptor_t topk_weights_desc,
                                  infiniopTensorDescriptor_t topk_ids_desc,
                                  float routed_scaling_factor) {
    Info info;
    CHECK_STATUS(createInfo(&info, router_logits_desc, input_ids_desc, tid2eid_desc, topk_weights_desc, topk_ids_desc, routed_scaling_factor));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, const void *router_logits, const void *input_ids, const void *tid2eid, void *topk_weights, void *topk_ids, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveHashTopk();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);

    const auto &i = _info;
    int device_id = this->device_id;
    std::array<int64_t, 2> logits_shape{static_cast<int64_t>(i.num_tokens), static_cast<int64_t>(i.num_experts)};
    std::array<int64_t, 2> logits_strides{static_cast<int64_t>(i.num_experts), 1};
    std::array<int64_t, 1> input_shape{static_cast<int64_t>(i.num_tokens)};
    std::array<int64_t, 1> input_strides{1};
    std::array<int64_t, 2> tid_shape{static_cast<int64_t>(i.num_tokens), static_cast<int64_t>(i.topk)};
    std::array<int64_t, 2> tid_strides{static_cast<int64_t>(i.topk), 1};
    std::array<int64_t, 2> out_shape{static_cast<int64_t>(i.num_tokens), static_cast<int64_t>(i.output_width)};
    std::array<int64_t, 2> out_strides{static_cast<int64_t>(i.output_width), 1};

    DLTensor logits_t, input_t, tid_t, weights_t, ids_t;
    auto logits_v = makeTensorView(router_logits, DLDataType{kDLFloat, 32, 1}, logits_shape, logits_strides, device_id, logits_t);
    auto input_v = makeTensorView(input_ids, DLDataType{kDLInt, 64, 1}, input_shape, input_strides, device_id, input_t);
    auto tid_v = makeTensorView(tid2eid, DLDataType{kDLInt, 32, 1}, tid_shape, tid_strides, device_id, tid_t);
    auto weights_v = makeTensorView(topk_weights, DLDataType{kDLFloat, 32, 1}, out_shape, out_strides, device_id, weights_t);
    auto ids_v = makeTensorView(topk_ids, DLDataType{kDLInt, 32, 1}, out_shape, out_strides, device_id, ids_t);

    fn(logits_v, input_v, tid_v, weights_v, ids_v, static_cast<double>(i.routed_scaling_factor));
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}

} // namespace op::dsv4_sglang_hash_topk::nvidia
