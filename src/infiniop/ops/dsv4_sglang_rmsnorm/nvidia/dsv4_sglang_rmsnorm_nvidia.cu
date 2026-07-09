#include "../../../handle.h"
#include "dsv4_sglang_rmsnorm_nvidia.cuh"

#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>

namespace {
using sglang_rmsnorm_fn_t = void (*)(
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    double);

constexpr const char *kSglangRmsnormSymbol = "_Z32sglang_rmsnorm_self_bf16_512_outN3tvm3ffi10TensorViewES1_d";

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

sglang_rmsnorm_fn_t resolveSglangRmsnorm() {
    static sglang_rmsnorm_fn_t fn = []() -> sglang_rmsnorm_fn_t {
        tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
        tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL);
        void *handle = tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            return nullptr;
        }
        return reinterpret_cast<sglang_rmsnorm_fn_t>(dlsym(handle, kSglangRmsnormSymbol));
    }();
    return fn;
}

tvm::ffi::TensorView makeBf16TensorView(const void *data,
                                        const std::array<int64_t, 3> &shape,
                                        const std::array<int64_t, 3> &strides,
                                        int device_id,
                                        DLTensor &tensor) {
    tensor.data = const_cast<void *>(data);
    tensor.device = DLDevice{kDLROCM, device_id};
    tensor.ndim = 3;
    tensor.dtype = DLDataType{kDLBfloat, 16, 1};
    tensor.shape = const_cast<int64_t *>(shape.data());
    tensor.strides = const_cast<int64_t *>(strides.data());
    tensor.byte_offset = 0;
    return tvm::ffi::TensorView(&tensor);
}

} // namespace
#endif

namespace op::dsv4_sglang_rmsnorm::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, double eps) {
    Info info;
    CHECK_STATUS(createInfo(&info, output_desc, input_desc, eps));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *output, const void *input, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveSglangRmsnorm();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);

    const auto &i = _info;
    int device_id = this->device_id;
    std::array<int64_t, 3> shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.tokens), 512};
    std::array<int64_t, 3> strides{static_cast<int64_t>(i.tokens * 512), 512, 1};

    DLTensor input_t, output_t;
    auto input_v = makeBf16TensorView(input, shape, strides, device_id, input_t);
    auto output_v = makeBf16TensorView(output, shape, strides, device_id, output_t);

    fn(input_v, output_v, i.eps);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}

} // namespace op::dsv4_sglang_rmsnorm::nvidia
