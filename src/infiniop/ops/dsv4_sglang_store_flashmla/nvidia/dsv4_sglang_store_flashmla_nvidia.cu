#include "../../../handle.h"
#include "dsv4_sglang_store_flashmla_nvidia.cuh"
#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>
namespace {
using fn_t = void (*)(tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView);
constexpr const char *kSymbol = "_Z37sglang_store_flashmla_bf16_i32_page64N3tvm3ffi10TensorViewES1_S1_";
const char *deepseekOpsPath() {
    const char *env = std::getenv("DEEPSEEK_V4_OPS_SO");
    return env && env[0] ? env : "libdeepseek_v4_ops.so";
}
void *tryDlopen(const char *path, int flags) {
    dlerror();
    return dlopen(path, flags);
}
fn_t resolveFn() {
    static fn_t fn = []() -> fn_t { tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL); tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL); void *h = tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL); return h ? reinterpret_cast<fn_t>(dlsym(h, kSymbol)) : nullptr; }();
    return fn;
}
template <size_t N>
tvm::ffi::TensorView makeView(const void *data, DLDataType dtype, const std::array<int64_t, N> &shape, const std::array<int64_t, N> &strides, int dev, DLTensor &t) {
    t.data = const_cast<void *>(data);
    t.device = DLDevice{kDLROCM, dev};
    t.ndim = N;
    t.dtype = dtype;
    t.shape = const_cast<int64_t *>(shape.data());
    t.strides = const_cast<int64_t *>(strides.data());
    t.byte_offset = 0;
    return tvm::ffi::TensorView(&t);
}
} // namespace
#endif
namespace op::dsv4_sglang_store_flashmla::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t cache_desc, infiniopTensorDescriptor_t indices_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, input_desc, cache_desc, indices_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, const void *input, void *cache, const void *indices, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveFn();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);
    const auto &i = _info;
    int dev = this->device_id;
    std::array<int64_t, 2> in_shape{static_cast<int64_t>(i.tokens), static_cast<int64_t>(i.head_dim)};
    std::array<int64_t, 2> in_strides{static_cast<int64_t>(i.head_dim), 1};
    std::array<int64_t, 2> cache_shape{static_cast<int64_t>(i.cache_rows), static_cast<int64_t>(i.cache_cols)};
    std::array<int64_t, 2> cache_strides{static_cast<int64_t>(i.cache_cols), 1};
    std::array<int64_t, 1> idx_shape{static_cast<int64_t>(i.tokens)};
    std::array<int64_t, 1> idx_strides{1};
    DLTensor in_t, cache_t, idx_t;
    auto in_v = makeView(input, DLDataType{kDLBfloat, 16, 1}, in_shape, in_strides, dev, in_t);
    auto cache_v = makeView(cache, DLDataType{kDLUInt, 8, 1}, cache_shape, cache_strides, dev, cache_t);
    auto idx_v = makeView(indices, DLDataType{kDLInt, 32, 1}, idx_shape, idx_strides, dev, idx_t);
    fn(in_v, cache_v, idx_v);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}
} // namespace op::dsv4_sglang_store_flashmla::nvidia
