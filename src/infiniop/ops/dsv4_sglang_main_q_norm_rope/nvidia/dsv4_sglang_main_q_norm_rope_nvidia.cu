#include "../../../handle.h"
#include "dsv4_sglang_main_q_norm_rope_nvidia.cuh"
#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>
namespace {
using fn_t = void (*)(tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, double);
constexpr const char *kSymbol = "_Z35sglang_main_q_norm_rope_bf16_512_64N3tvm3ffi10TensorViewES1_S1_S1_d";
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
namespace op::dsv4_sglang_main_q_norm_rope::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t freqs_desc, infiniopTensorDescriptor_t positions_desc, double eps) {
    Info info;
    CHECK_STATUS(createInfo(&info, output_desc, input_desc, freqs_desc, positions_desc, eps));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, void *output, const void *input, const void *freqs, const void *positions, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveFn();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);
    const auto &i = _info;
    int dev = this->device_id;

    std::array<int64_t, 3> q_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.heads), 512};
    std::array<int64_t, 3> q_strides{static_cast<int64_t>(i.heads * 512), 512, 1};
    std::array<int64_t, 1> pos_shape{static_cast<int64_t>(i.batch)};
    std::array<int64_t, 1> pos_strides{1};
    std::array<int64_t, 2> freq_shape{static_cast<int64_t>(i.freqs_rows), 64};
    std::array<int64_t, 2> freq_strides{64, 1};
    DLTensor in_t, out_t, freq_t, pos_t;
    auto in_v = makeView(input, DLDataType{kDLBfloat, 16, 1}, q_shape, q_strides, dev, in_t);
    auto out_v = makeView(output, DLDataType{kDLBfloat, 16, 1}, q_shape, q_strides, dev, out_t);
    auto freq_v = makeView(freqs, DLDataType{kDLFloat, 32, 1}, freq_shape, freq_strides, dev, freq_t);
    auto pos_v = makeView(positions, DLDataType{kDLInt, 32, 1}, pos_shape, pos_strides, dev, pos_t);
    fn(in_v, out_v, freq_v, pos_v, i.eps);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}
} // namespace op::dsv4_sglang_main_q_norm_rope::nvidia
