#include "../../../handle.h"
#include "dsv4_sglang_silu_and_mul_clamp_nvidia.cuh"
#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>
namespace {
using fn_t = void (*)(tvm::ffi::TensorView, tvm::ffi::TensorView, double);
constexpr const char *kSymbol = "_Z30sglang_silu_and_mul_clamp_bf16N3tvm3ffi10TensorViewES1_d";
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
tvm::ffi::TensorView makeView(const void *data, const std::array<int64_t, 2> &shape, const std::array<int64_t, 2> &strides, int dev, DLTensor &t) {
    t.data = const_cast<void *>(data);
    t.device = DLDevice{kDLROCM, dev};
    t.ndim = 2;
    t.dtype = DLDataType{kDLBfloat, 16, 1};
    t.shape = const_cast<int64_t *>(shape.data());
    t.strides = const_cast<int64_t *>(strides.data());
    t.byte_offset = 0;
    return tvm::ffi::TensorView(&t);
}
} // namespace
#endif
namespace op::dsv4_sglang_silu_and_mul_clamp::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, double limit) {
    Info info;
    CHECK_STATUS(createInfo(&info, output_desc, input_desc, limit));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, void *output, const void *input, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveFn();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);
    const auto &i = _info;
    int dev = this->device_id;
    std::array<int64_t, 2> in_shape{static_cast<int64_t>(i.tokens), static_cast<int64_t>(i.hidden * 2)};
    std::array<int64_t, 2> in_strides{static_cast<int64_t>(i.hidden * 2), 1};
    std::array<int64_t, 2> out_shape{static_cast<int64_t>(i.tokens), static_cast<int64_t>(i.hidden)};
    std::array<int64_t, 2> out_strides{static_cast<int64_t>(i.hidden), 1};
    DLTensor in_t, out_t;
    auto in_v = makeView(input, in_shape, in_strides, dev, in_t);
    auto out_v = makeView(output, out_shape, out_strides, dev, out_t);
    fn(in_v, out_v, i.limit);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}
} // namespace op::dsv4_sglang_silu_and_mul_clamp::nvidia
