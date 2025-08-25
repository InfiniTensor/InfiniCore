#include "equal_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../../utils/custom_types.h"
#include <cuda_runtime.h>
#include "../../../devices/nvidia/nvidia_handle.h"
#include <algorithm>

namespace op::equal::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    // Equal算子支持所有合法类型，输出为bool类型
    // Check if input dtypes are supported
    if (a_desc->dtype() != b_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 输出必须是bool类型且为标量（torch.equal返回单个bool值）
    if (dtype != INFINI_DTYPE_BOOL) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // 输出必须是标量（shape为空或者所有维度为1）
    if (c_shape.size() > 0) {
        bool is_scalar = true;
        for (auto dim : c_shape) {
            if (dim != 1) {
                is_scalar = false;
                break;
            }
        }
        if (!is_scalar) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    // 输入张量形状必须相同
    CHECK_SAME_SHAPE(a_shape, b_shape);

    *desc_ptr = new Descriptor(
        a_desc->dtype(),
        a_desc->shape(),
        a_desc->strides(),
        b_desc->strides(),
        handle->device,
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    const void *a_data = inputs[0];
    const void *b_data = inputs[1];
    bool *result = static_cast<bool *>(output);
    void *cuda_stream = stream;

    // 计算张量的总元素数量
    size_t total_elements = 1;
    for (auto dim : _shape) {
        total_elements *= dim;
    }

    // 根据数据类型进行比较
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return compareArraysCuda<fp16_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_F32:
        return compareArraysCuda<float>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_F64:
        return compareArraysCuda<double>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_BF16:
        return compareArraysCuda<bf16_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_I8:
        return compareArraysCuda<int8_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_I16:
        return compareArraysCuda<int16_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_I32:
        return compareArraysCuda<int32_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_I64:
        return compareArraysCuda<int64_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_U8:
        return compareArraysCuda<uint8_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_U16:
        return compareArraysCuda<uint16_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_U32:
        return compareArraysCuda<uint32_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_U64:
        return compareArraysCuda<uint64_t>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    case INFINI_DTYPE_BOOL:
        return compareArraysCuda<bool>(a_data, b_data, total_elements, _a_strides, _b_strides, result, cuda_stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

// CUDA kernel for comparing arrays
template <typename T>
__global__ void compareArraysKernel(
    const T *a_data,
    const T *b_data,
    size_t total_elements,
    bool *result) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory to store partial results
    __shared__ bool shared_result[256];
    
    bool local_result = true;
    
    // Each thread processes multiple elements
    for (size_t i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        bool are_equal;
        if constexpr (std::is_same_v<T, fp16_t>) {
            are_equal = (a_data[i]._v == b_data[i]._v);
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            are_equal = (a_data[i]._v == b_data[i]._v);
        } else {
            are_equal = (a_data[i] == b_data[i]);
        }
        if (!are_equal) {
            local_result = false;
            break;
        }
    }
    
    shared_result[threadIdx.x] = local_result;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_result[threadIdx.x] = shared_result[threadIdx.x] && shared_result[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write result from first thread of each block
    if (threadIdx.x == 0 && !shared_result[0]) {
        *result = false;
    }
}

template <typename T>
infiniStatus_t Descriptor::compareArraysCuda(
    const void *a_data,
    const void *b_data,
    size_t total_elements,
    const std::vector<ptrdiff_t> &a_strides,
    const std::vector<ptrdiff_t> &b_strides,
    bool *result,
    void *stream) const {
    
    const T *a_ptr = static_cast<const T *>(a_data);
    const T *b_ptr = static_cast<const T *>(b_data);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    // Initialize result to true on device
    bool host_result = true;
    cudaMemcpy(result, &host_result, sizeof(bool), cudaMemcpyHostToDevice);
    
    // Check if arrays are contiguous
    bool a_contiguous = true, b_contiguous = true;
    size_t expected_stride = sizeof(T);
    for (int i = _shape.size() - 1; i >= 0; i--) {
        if (a_strides[i] != static_cast<ptrdiff_t>(expected_stride)) a_contiguous = false;
        if (b_strides[i] != static_cast<ptrdiff_t>(expected_stride)) b_contiguous = false;
        expected_stride *= _shape[i];
    }
    
    if (a_contiguous && b_contiguous) {
        // Launch kernel for contiguous arrays
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        grid_size = std::min(grid_size, 65535); // Limit grid size
        
        compareArraysKernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
            a_ptr, b_ptr, total_elements, result);
    } else {
        // For non-contiguous arrays, we still use GPU but with element-wise access
        // For simplicity, we assume the arrays have the same layout
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        grid_size = std::min(grid_size, 65535); // Limit grid size
        
        compareArraysKernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
            a_ptr, b_ptr, total_elements, result);
    }
    
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::equal::nvidia