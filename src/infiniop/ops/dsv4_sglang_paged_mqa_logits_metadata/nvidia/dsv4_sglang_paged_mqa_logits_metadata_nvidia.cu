#include "../../../handle.h"
#include "dsv4_sglang_paged_mqa_logits_metadata_nvidia.cuh"
#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>
namespace {
using fn_t = void (*)(tvm::ffi::TensorView, tvm::ffi::TensorView);
constexpr const char *kSymbol = "_Z32sglang_paged_mqa_logits_metadataN3tvm3ffi10TensorViewES1_";
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
tvm::ffi::TensorView makeView(const void *data, const std::array<int64_t, N> &shape, const std::array<int64_t, N> &strides, int dev, DLTensor &t) {
    t.data = const_cast<void *>(data);
    t.device = DLDevice{kDLROCM, dev};
    t.ndim = N;
    t.dtype = DLDataType{kDLInt, 32, 1};
    t.shape = const_cast<int64_t *>(shape.data());
    t.strides = const_cast<int64_t *>(strides.data());
    t.byte_offset = 0;
    return tvm::ffi::TensorView(&t);
}
} // namespace
#endif
namespace op::dsv4_sglang_paged_mqa_logits_metadata::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t metadata_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, seq_lens_desc, metadata_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, const void *seq_lens, void *metadata, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveFn();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);
    const auto &i = _info;
    int dev = this->device_id;
    std::array<int64_t, 1> seq_shape{static_cast<int64_t>(i.batch)};
    std::array<int64_t, 1> seq_strides{1};
    std::array<int64_t, 2> meta_shape{static_cast<int64_t>(i.metadata_rows), 2};
    std::array<int64_t, 2> meta_strides{2, 1};
    DLTensor seq_t, meta_t;
    auto seq_v = makeView(seq_lens, seq_shape, seq_strides, dev, seq_t);
    auto meta_v = makeView(metadata, meta_shape, meta_strides, dev, meta_t);
    fn(seq_v, meta_v);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}
} // namespace op::dsv4_sglang_paged_mqa_logits_metadata::nvidia
