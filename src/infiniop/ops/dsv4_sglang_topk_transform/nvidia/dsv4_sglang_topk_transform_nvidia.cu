#include "../../../handle.h"
#include "dsv4_sglang_topk_transform_nvidia.cuh"

#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <optional>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

namespace {
using topk_transform_fn_t = void (*)(
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    long,
    tvm::ffi::Optional<tvm::ffi::TensorView>);

constexpr const char *kTopk512Symbol = "_Z25sglang_topk_transform_512N3tvm3ffi10TensorViewES1_S1_S1_lNS0_8OptionalIS1_vEE";
constexpr const char *kTopk1024Symbol = "_Z26sglang_topk_transform_1024N3tvm3ffi10TensorViewES1_S1_S1_lNS0_8OptionalIS1_vEE";

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

topk_transform_fn_t resolveTopkTransform(size_t width) {
    static void *handle = []() -> void * {
        tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
        tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL);
        return tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL);
    }();
    if (!handle) {
        return nullptr;
    }
    const char *symbol = width == 512 ? kTopk512Symbol : kTopk1024Symbol;
    return reinterpret_cast<topk_transform_fn_t>(dlsym(handle, symbol));
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

namespace op::dsv4_sglang_topk_transform::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_table_desc, infiniopTensorDescriptor_t page_indices_desc, infiniopTensorDescriptor_t raw_indices_desc, int64_t page_size) {
    Info info;
    CHECK_STATUS(createInfo(&info, scores_desc, seq_lens_desc, page_table_desc, page_indices_desc, raw_indices_desc, page_size));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, const void *scores, const void *seq_lens, const void *page_table, void *page_indices, void *raw_indices, void *) const {
#if defined(ENABLE_HYGON_API)
    const auto &i = _info;
    auto fn = resolveTopkTransform(i.topk_width);
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);

    int device_id = this->device_id;
    std::array<int64_t, 2> scores_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.scores_width)};
    std::array<int64_t, 2> scores_strides{static_cast<int64_t>(i.scores_width), 1};
    std::array<int64_t, 1> seq_shape{static_cast<int64_t>(i.batch)};
    std::array<int64_t, 1> seq_strides{1};
    std::array<int64_t, 2> table_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.pages)};
    std::array<int64_t, 2> table_strides{static_cast<int64_t>(i.pages), 1};
    std::array<int64_t, 2> out_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.topk_width)};
    std::array<int64_t, 2> out_strides{static_cast<int64_t>(i.topk_width), 1};

    DLTensor scores_t, seq_t, table_t, page_t, raw_t;
    auto scores_v = makeTensorView(scores, DLDataType{kDLFloat, 32, 1}, scores_shape, scores_strides, device_id, scores_t);
    auto seq_v = makeTensorView(seq_lens, DLDataType{kDLInt, 32, 1}, seq_shape, seq_strides, device_id, seq_t);
    auto table_v = makeTensorView(page_table, DLDataType{kDLInt, 32, 1}, table_shape, table_strides, device_id, table_t);
    auto page_v = makeTensorView(page_indices, DLDataType{kDLInt, 32, 1}, out_shape, out_strides, device_id, page_t);
    auto raw_v = makeTensorView(raw_indices, DLDataType{kDLInt, 32, 1}, out_shape, out_strides, device_id, raw_t);
    tvm::ffi::Optional<tvm::ffi::TensorView> raw_opt(raw_v);

    fn(scores_v, seq_v, table_v, page_v, static_cast<long>(i.page_size), raw_opt);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}

} // namespace op::dsv4_sglang_topk_transform::nvidia
