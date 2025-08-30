#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "leaky_relu_nv.cuh"

// Device conversion functions for fp16_t
__device__ __forceinline__ float device_f16_to_f32(fp16_t val) {
    // Convert custom fp16_t to CUDA half using reinterpret_cast, then to float
    __half h = *reinterpret_cast<const __half*>(&val._v);
    return __half2float(h);
}

__device__ __forceinline__ fp16_t device_f32_to_f16(float val) {
    // Convert float to CUDA half, then to custom fp16_t
    __half h = __float2half(val);
    return fp16_t{*reinterpret_cast<const uint16_t*>(&h)};
}

// Device conversion functions for bf16_t
__device__ __forceinline__ float device_bf16_to_f32(bf16_t val) {
    // bf16 to f32: put bf16 bits in high 16 bits of f32, low 16 bits are 0
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;
    float result;
    memcpy(&result, &bits32, sizeof(result));
    return result;
}

__device__ __forceinline__ bf16_t device_f32_to_bf16(float val) {
    // f32 to bf16: round-to-nearest-even truncation
    uint32_t bits32;
    memcpy(&bits32, &val, sizeof(bits32));
    const uint32_t rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1);
    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);
    return bf16_t{bf16_bits};
}

namespace op::leaky_relu::cuda {

// Function to set negative slope
void setNegativeSlope(float slope) {
    cudaMemcpyToSymbol(g_negative_slope, &slope, sizeof(float));
}

}

namespace op::leaky_relu::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    std::vector<infiniopTensorDescriptor_t> input_descs,
    float negative_slope) {
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = output_desc->dtype();
    
    if (input_descs.size() != 1) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto input_desc = input_descs[0];
    
    // Check data type compatibility
    if (output_desc->dtype() != input_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    const auto &y_shape = output_desc->shape();
    const auto &x_shape = input_desc->shape();
    
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    CHECK_SAME_SHAPE(y_shape, x_shape);
    
    // Set the negative slope in device constant memory
    op::leaky_relu::cuda::setNegativeSlope(negative_slope);
    
    // Create CUDA elementwise descriptor
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, output_desc, input_descs);
    
    // Store negative slope in descriptor
    reinterpret_cast<Descriptor*>(*desc_ptr)->_negative_slope = negative_slope;
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {
    
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    
    // Set the negative slope before calculation
    op::leaky_relu::cuda::setNegativeSlope(_negative_slope);
    
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, op::leaky_relu::cuda::LeakyReLUOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, op::leaky_relu::cuda::LeakyReLUOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, op::leaky_relu::cuda::LeakyReLUOp, __nv_bfloat16>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

}