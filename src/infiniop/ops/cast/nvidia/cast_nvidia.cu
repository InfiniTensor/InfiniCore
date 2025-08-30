#include "cast_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_handle.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../cuda/kernel.cuh"
#include "../../../../utils/custom_types.h"

// Device versions of fp16 conversion functions
__device__ __forceinline__ float device_f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;
    if (exponent == 31) {
        if (mantissa != 0) {
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
            f32 = sign | 0x7F800000;
        }
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    return __uint_as_float(f32);
}

__device__ __forceinline__ fp16_t device_f32_to_f16(float val) {
    uint32_t f32 = __float_as_uint(val);
    uint16_t sign = (f32 >> 16) & 0x8000;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x7FFFFF;

    if (exponent >= 16) {
        if (exponent == 128 && mantissa != 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7E00)};
        }
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00)};
    } else if (exponent >= -14) {
        return fp16_t{(uint16_t)(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    } else if (exponent >= -24) {
        mantissa |= 0x800000;
        mantissa >>= (-14 - exponent);
        return fp16_t{(uint16_t)(sign | (mantissa >> 13))};
    } else {
        return fp16_t{(uint16_t)sign};
    }
}

namespace op::cast::nvidia {

struct Descriptor::Opaque {
    size_t numel;
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::Descriptor(infiniDtype_t input_dtype, infiniDtype_t output_dtype, size_t workspace_size)
    : InfiniopDescriptor{INFINI_DEVICE_NVIDIA, static_cast<int>(workspace_size)}, 
      _input_dtype(input_dtype), 
      _output_dtype(output_dtype),
      _workspace_size(workspace_size) {
    _opaque = new Opaque();
}

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto input_dtype = input_desc_vec[0]->dtype();
    auto output_dtype = output_desc->dtype();

    // 检查支持的类型转换
    bool valid_cast = false;
    
    // 整数类型之间的转换
    if ((input_dtype == INFINI_DTYPE_I32 || input_dtype == INFINI_DTYPE_I64 || 
         input_dtype == INFINI_DTYPE_U32 || input_dtype == INFINI_DTYPE_U64) &&
        (output_dtype == INFINI_DTYPE_I32 || output_dtype == INFINI_DTYPE_I64 || 
         output_dtype == INFINI_DTYPE_U32 || output_dtype == INFINI_DTYPE_U64)) {
        valid_cast = true;
    }
    
    // 浮点类型之间的转换
    if ((input_dtype == INFINI_DTYPE_F64 || input_dtype == INFINI_DTYPE_F32 || input_dtype == INFINI_DTYPE_F16) &&
        (output_dtype == INFINI_DTYPE_F64 || output_dtype == INFINI_DTYPE_F32 || output_dtype == INFINI_DTYPE_F16)) {
        valid_cast = true;
    }
    
    // 整数类型转浮点类型
    if ((input_dtype == INFINI_DTYPE_I32 || input_dtype == INFINI_DTYPE_I64 || 
         input_dtype == INFINI_DTYPE_U32 || input_dtype == INFINI_DTYPE_U64) &&
        (output_dtype == INFINI_DTYPE_F64 || output_dtype == INFINI_DTYPE_F32 || output_dtype == INFINI_DTYPE_F16)) {
        valid_cast = true;
    }
    
    // 浮点类型转整数类型
    if ((input_dtype == INFINI_DTYPE_F64 || input_dtype == INFINI_DTYPE_F32 || input_dtype == INFINI_DTYPE_F16) &&
        (output_dtype == INFINI_DTYPE_I32 || output_dtype == INFINI_DTYPE_I64 || 
         output_dtype == INFINI_DTYPE_U32 || output_dtype == INFINI_DTYPE_U64)) {
        valid_cast = true;
    }
    
    if (!valid_cast) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // 检查形状一致性
    const auto &input_shape = input_desc_vec[0]->shape();
    const auto &output_shape = output_desc->shape();
    if (input_shape != output_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto desc = new Descriptor(input_dtype, output_dtype, 0);
    desc->_opaque->numel = output_desc->numel();
    desc->_opaque->internal = handle->internal();
    
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::workspaceSize() const {
    return _workspace_size;
}

// Device-side cast function
template <typename Tout, typename Tin>
__device__ __forceinline__ Tout device_cast(const Tin &value) {
    if constexpr (std::is_same_v<Tin, fp16_t> && std::is_same_v<Tout, float>) {
        return device_f16_to_f32(value);
    } else if constexpr (std::is_same_v<Tin, float> && std::is_same_v<Tout, fp16_t>) {
        return device_f32_to_f16(value);
    } else if constexpr (std::is_same_v<Tin, fp16_t> && std::is_same_v<Tout, double>) {
        return static_cast<double>(device_f16_to_f32(value));
    } else if constexpr (std::is_same_v<Tin, double> && std::is_same_v<Tout, fp16_t>) {
        return device_f32_to_f16(static_cast<float>(value));
    } else if constexpr (std::is_same_v<Tout, fp16_t>) {
        // Convert any other type to fp16_t via float
        return device_f32_to_f16(static_cast<float>(value));
    } else if constexpr (std::is_same_v<Tin, fp16_t>) {
        // Convert fp16_t to any other type via float
        return static_cast<Tout>(device_f16_to_f32(value));
    } else {
        return static_cast<Tout>(value);
    }
}

// CUDA kernel for cast operation
template <typename Tin, typename Tout>
__global__ void castKernel(const Tin *input, Tout *output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = device_cast<Tout>(input[idx]);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    const void *input = inputs[0];
    size_t numel = _opaque->numel;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // 计算grid和block大小
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 根据输入和输出数据类型进行转换
    if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_I64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int32_t*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_I32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int64_t*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_U64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint32_t*>(input), static_cast<uint64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_U32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint64_t*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_U32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int32_t*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_I32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint32_t*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_U64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int64_t*>(input), static_cast<uint64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_I64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint64_t*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_F64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const float*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_F32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const double*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_F32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const fp16_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_F16) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const float*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_F64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const fp16_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_F16) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const double*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_F32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int32_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_F64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int32_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_F16) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int32_t*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_F32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int64_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_F64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int64_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_F16) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const int64_t*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_F32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint32_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_F64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint32_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_F16) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint32_t*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_F32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint64_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_F64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint64_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_F16) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const uint64_t*>(input), static_cast<fp16_t*>(output), numel);
    // 浮点数到整数的转换
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_I32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const float*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_I64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const float*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_U32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const float*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_U64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const float*>(input), static_cast<uint64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_I32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const double*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_I64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const double*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_U32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const double*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_U64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const double*>(input), static_cast<uint64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_I32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const fp16_t*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_I64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const fp16_t*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_U32) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const fp16_t*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_U64) {
        castKernel<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            static_cast<const fp16_t*>(input), static_cast<uint64_t*>(output), numel);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // 检查CUDA错误
    CHECK_OR_RETURN(cudaGetLastError() == cudaSuccess, INFINI_STATUS_INTERNAL_ERROR);
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::cast::nvidia