#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_linear_bf16_fp32_nvidia.cuh"

#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>
#endif

namespace {

template <typename T>
__global__ void castToFloatKernel(const T *x, float *x_fp32, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        x_fp32[idx] = static_cast<float>(x[idx]);
    }
}

template <typename T>
infiniStatus_t castToFloat(const void *x, void *workspace, size_t total, cudaStream_t stream) {
    constexpr int block = 256;
    size_t blocks = (total + block - 1) / block;
    int grid = static_cast<int>(blocks < 4096 ? blocks : 4096);
    castToFloatKernel<T><<<grid, block, 0, stream>>>(static_cast<const T *>(x), static_cast<float *>(workspace), total);
    return INFINI_STATUS_SUCCESS;
}

#if defined(ENABLE_HYGON_API)
using linear_bf16_fp32_fn_t = void (*)(
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView);

constexpr const char *kLinearBf16Fp32Symbol = "_Z20linear_bf16_fp32_outN3tvm3ffi10TensorViewES1_S1_";

bool enableTonySo() {
    const char *env = std::getenv("INFINICORE_DSV4_ENABLE_TONY_SO");
    return env && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0);
}

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

linear_bf16_fp32_fn_t resolveLinearBf16Fp32() {
    static linear_bf16_fp32_fn_t fn = []() -> linear_bf16_fp32_fn_t {
        tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
        tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL);
        void *handle = tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            return nullptr;
        }
        return reinterpret_cast<linear_bf16_fp32_fn_t>(dlsym(handle, kLinearBf16Fp32Symbol));
    }();
    return fn;
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
                                    const std::array<int64_t, 2> &shape,
                                    const std::array<int64_t, 2> &strides,
                                    int device_id,
                                    DLTensor &tensor) {
    return makeTensorView(const_cast<void *>(data), dtype, shape, strides, device_id, tensor);
}

infiniStatus_t tryLinearBf16Fp32So(const op::dsv4_linear_bf16_fp32::Info &info,
                                   int device_id,
                                   void *y,
                                   const void *x,
                                   const void *w,
                                   bool *called) {
    *called = false;
    if (info.x_dtype != INFINI_DTYPE_BF16 || !enableTonySo()) {
        return INFINI_STATUS_SUCCESS;
    }
    auto fn = resolveLinearBf16Fp32();
    if (!fn) {
        return INFINI_STATUS_SUCCESS;
    }

    std::array<int64_t, 2> x_shape{static_cast<int64_t>(info.m), static_cast<int64_t>(info.k)};
    std::array<int64_t, 2> x_strides{static_cast<int64_t>(info.k), 1};
    std::array<int64_t, 2> w_shape{static_cast<int64_t>(info.n), static_cast<int64_t>(info.k)};
    std::array<int64_t, 2> w_strides{static_cast<int64_t>(info.k), 1};
    std::array<int64_t, 2> y_shape{static_cast<int64_t>(info.m), static_cast<int64_t>(info.n)};
    std::array<int64_t, 2> y_strides{static_cast<int64_t>(info.n), 1};

    DLTensor x_t, w_t, y_t;
    auto x_v = makeTensorView(x, DLDataType{kDLBfloat, 16, 1}, x_shape, x_strides, device_id, x_t);
    auto w_v = makeTensorView(w, DLDataType{kDLFloat, 32, 1}, w_shape, w_strides, device_id, w_t);
    auto y_v = makeTensorView(y, DLDataType{kDLFloat, 32, 1}, y_shape, y_strides, device_id, y_t);
    fn(x_v, w_v, y_v);
    *called = true;
    return INFINI_STATUS_SUCCESS;
}
#endif

} // namespace

namespace op::dsv4_linear_bf16_fp32::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    Info info;
    CHECK_STATUS(createInfo(&info, y_desc, x_desc, w_desc));
    size_t workspace_size = info.m * info.k * sizeof(float);
    *desc_ptr = new Descriptor(info, workspace_size, new Opaque{handle->internal()}, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size, void *y, const void *x, const void *w, void *stream) const {
#if defined(ENABLE_HYGON_API)
    bool called_so = false;
    CHECK_STATUS(tryLinearBf16Fp32So(_info, this->device_id, y, x, w, &called_so));
    if (called_so) {
        return INFINI_STATUS_SUCCESS;
    }
#endif

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    size_t x_numel = _info.m * _info.k;
    switch (_info.x_dtype) {
    case INFINI_DTYPE_F16:
        CHECK_STATUS(castToFloat<half>(x, workspace, x_numel, cuda_stream));
        break;
    case INFINI_DTYPE_BF16:
        CHECK_STATUS(castToFloat<__nv_bfloat16>(x, workspace, x_numel, cuda_stream));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    cudaDataType compute_type = CUDA_R_32F;
#else
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#endif

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int m = static_cast<int>(_info.m);
    const int n = static_cast<int>(_info.n);
    const int k = static_cast<int>(_info.k);
    const float *x_fp32 = static_cast<const float *>(workspace);

    CHECK_STATUS(_opaque->internal->useCublas(
        cuda_stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha,
                w,
                CUDA_R_32F,
                k,
                x_fp32,
                CUDA_R_32F,
                k,
                &beta,
                y,
                CUDA_R_32F,
                n,
                compute_type,
                CUBLAS_GEMM_DEFAULT));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_linear_bf16_fp32::nvidia
