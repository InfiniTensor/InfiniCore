#include "../../../handle.h"
#include "dsv4_sglang_mega_moe_pre_dispatch_nvidia.cuh"
#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>
namespace {
using fn_t = void (*)(tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView);
constexpr const char *kSymbol = "_Z32sglang_mega_moe_pre_dispatch_g32N3tvm3ffi10TensorViewES1_S1_S1_S1_S1_S1_";
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
namespace op::dsv4_sglang_mega_moe_pre_dispatch::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t topk_idx_desc, infiniopTensorDescriptor_t topk_weights_desc, infiniopTensorDescriptor_t buf_x_desc, infiniopTensorDescriptor_t buf_x_sf_desc, infiniopTensorDescriptor_t buf_topk_idx_desc, infiniopTensorDescriptor_t buf_topk_weights_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, x_desc, topk_idx_desc, topk_weights_desc, buf_x_desc, buf_x_sf_desc, buf_topk_idx_desc, buf_topk_weights_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, const void *x, const void *topk_idx, const void *topk_weights, void *buf_x, void *buf_x_sf, void *buf_topk_idx, void *buf_topk_weights, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveFn();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);
    const auto &i = _info;
    int dev = this->device_id;
    std::array<int64_t, 2> x_shape{static_cast<int64_t>(i.tokens), static_cast<int64_t>(i.hidden)};
    std::array<int64_t, 2> x_strides{static_cast<int64_t>(i.hidden), 1};
    std::array<int64_t, 2> topk_shape{static_cast<int64_t>(i.tokens), static_cast<int64_t>(i.topk)};
    std::array<int64_t, 2> topk_strides{static_cast<int64_t>(i.topk), 1};
    std::array<int64_t, 2> buf_shape{static_cast<int64_t>(i.padded), static_cast<int64_t>(i.hidden)};
    std::array<int64_t, 2> buf_strides{static_cast<int64_t>(i.hidden), 1};
    std::array<int64_t, 2> sf_shape{static_cast<int64_t>(i.padded), static_cast<int64_t>(i.sf_cols)};
    std::array<int64_t, 2> sf_strides{static_cast<int64_t>(i.sf_cols), 1};
    std::array<int64_t, 2> buf_topk_shape{static_cast<int64_t>(i.padded), static_cast<int64_t>(i.topk)};
    std::array<int64_t, 2> buf_topk_strides{static_cast<int64_t>(i.topk), 1};
    DLTensor x_t, idx_t, w_t, bx_t, sf_t, bidx_t, bw_t;
    auto x_v = makeView(x, DLDataType{kDLBfloat, 16, 1}, x_shape, x_strides, dev, x_t);
    auto idx_v = makeView(topk_idx, DLDataType{kDLInt, 32, 1}, topk_shape, topk_strides, dev, idx_t);
    auto w_v = makeView(topk_weights, DLDataType{kDLFloat, 32, 1}, topk_shape, topk_strides, dev, w_t);
    auto bx_v = makeView(buf_x, DLDataType{kDLInt, 8, 1}, buf_shape, buf_strides, dev, bx_t);
    auto sf_v = makeView(buf_x_sf, DLDataType{kDLInt, 32, 1}, sf_shape, sf_strides, dev, sf_t);
    auto bidx_v = makeView(buf_topk_idx, DLDataType{kDLInt, 64, 1}, buf_topk_shape, buf_topk_strides, dev, bidx_t);
    auto bw_v = makeView(buf_topk_weights, DLDataType{kDLFloat, 32, 1}, buf_topk_shape, buf_topk_strides, dev, bw_t);
    fn(x_v, idx_v, w_v, bx_v, sf_v, bidx_v, bw_v);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}
} // namespace op::dsv4_sglang_mega_moe_pre_dispatch::nvidia
