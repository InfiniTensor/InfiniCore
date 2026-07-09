#include "../../../handle.h"
#include "dsv4_sglang_compress_c128_online_v2_prefill_nvidia.cuh"
#if defined(ENABLE_HYGON_API)
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>
namespace {
using fn_t = void (*)(tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView);
constexpr const char *kSymbol = "_Z42sglang_compress_c128_online_v2_prefill_512N3tvm3ffi10TensorViewES1_S1_S1_S1_S1_";
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
DLDataType toDlDtype(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F32:
        return DLDataType{kDLFloat, 32, 1};
    case INFINI_DTYPE_F16:
        return DLDataType{kDLFloat, 16, 1};
    case INFINI_DTYPE_BF16:
        return DLDataType{kDLBfloat, 16, 1};
    case INFINI_DTYPE_I64:
        return DLDataType{kDLInt, 64, 1};
    case INFINI_DTYPE_I32:
        return DLDataType{kDLInt, 32, 1};
    case INFINI_DTYPE_I8:
        return DLDataType{kDLInt, 8, 1};
    case INFINI_DTYPE_U8:
        return DLDataType{kDLUInt, 8, 1};
    case INFINI_DTYPE_F8:
        return DLDataType{kDLFloat, 8, 1};
    default:
        return DLDataType{kDLUInt, 8, 1};
    }
}
tvm::ffi::TensorView makeView(const void *data, const op::dsv4_sglang_compress_c128_online_v2_prefill::TensorInfo &info, int dev, DLTensor &t) {
    t.data = const_cast<void *>(data);
    t.device = DLDevice{kDLROCM, dev};
    t.ndim = static_cast<int>(info.shape.size());
    t.dtype = toDlDtype(info.dtype);
    t.shape = const_cast<int64_t *>(info.shape.data());
    t.strides = const_cast<int64_t *>(info.strides.data());
    t.byte_offset = 0;
    return tvm::ffi::TensorView(&t);
}
} // namespace
#endif
namespace op::dsv4_sglang_compress_c128_online_v2_prefill::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t kv_score_buffer_desc, infiniopTensorDescriptor_t kv_score_input_desc, infiniopTensorDescriptor_t kv_output_desc, infiniopTensorDescriptor_t ape_desc, infiniopTensorDescriptor_t plan_c_desc, infiniopTensorDescriptor_t plan_w_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, kv_score_buffer_desc, kv_score_input_desc, kv_output_desc, ape_desc, plan_c_desc, plan_w_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, const void *kv_score_buffer, const void *kv_score_input, void *kv_output, const void *ape, const void *plan_c, const void *plan_w, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveFn();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);
    int dev = this->device_id;
    DLTensor kv_score_buffer_t;
    auto kv_score_buffer_v = makeView(kv_score_buffer, _info.tensors[0], dev, kv_score_buffer_t);
    DLTensor kv_score_input_t;
    auto kv_score_input_v = makeView(kv_score_input, _info.tensors[1], dev, kv_score_input_t);
    DLTensor kv_output_t;
    auto kv_output_v = makeView(kv_output, _info.tensors[2], dev, kv_output_t);
    DLTensor ape_t;
    auto ape_v = makeView(ape, _info.tensors[3], dev, ape_t);
    DLTensor plan_c_t;
    auto plan_c_v = makeView(plan_c, _info.tensors[4], dev, plan_c_t);
    DLTensor plan_w_t;
    auto plan_w_v = makeView(plan_w, _info.tensors[5], dev, plan_w_t);
    fn(kv_score_buffer_v, kv_score_input_v, kv_output_v, ape_v, plan_c_v, plan_w_v);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}
} // namespace op::dsv4_sglang_compress_c128_online_v2_prefill::nvidia
