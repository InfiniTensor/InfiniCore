// #include "../../../devices/nvidia/nvidia_common.cuh"
// #include "../../../devices/nvidia/nvidia_handle.cuh"
// #include "../../../devices/nvidia/nvidia_kernel_common.cuh"

// #include "../cuda/kernel.cuh"
// #include "../info.h"
// #include "pixel_shuffle_nvidia.cuh"

// namespace op::pixel_shuffle::nvidia {

// template <typename Tdata>
// infiniStatus_t calculate_pixel_shuffle(
//     const PixelShuffleInfo &info,
//     Tdata *output,
//     const Tdata *input,
//     cudaStream_t stream,
//     void *workspace) {
    
//     // No workspace needed for the new kernel
//     (void)workspace;
    
//     // Calculate grid and block dimensions
//     size_t total_elements = (size_t)info.B * info.C_out * info.H_out * info.W_out;
//     dim3 blockSize(256);
//     dim3 gridSize((total_elements + blockSize.x - 1) / blockSize.x);
    
//     // 直接使用 T* 指针和元素 stride，与其他 CUDA 算子保持一致
//     pixel_shuffle_kernel<Tdata><<<gridSize, blockSize, 0, stream>>>(
//         input,
//         output,
//         info.r,
//         info.B,
//         info.C_out,
//         info.H_out,
//         info.W_out,
//         info.input_b_stride,
//         info.input_c_stride,
//         info.input_h_stride,
//         info.input_w_stride,
//         info.output_b_stride,
//         info.output_c_stride,
//         info.output_h_stride,
//         info.output_w_stride
//     );
    
//     CHECK_CUDA(cudaGetLastError());
//     return INFINI_STATUS_SUCCESS;
// }

// struct Descriptor::Opaque {
//     std::shared_ptr<device::nvidia::Handle::Internal> internal;
// };

// Descriptor::~Descriptor() {
//     delete _opaque;
// }

// infiniStatus_t Descriptor::create(
//     infiniopHandle_t handle_,
//     Descriptor **desc_ptr,
//     infiniopTensorDescriptor_t output_desc,
//     infiniopTensorDescriptor_t input_desc,
//     int upscale_factor) {
    
//     auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    
//     auto dtype = output_desc->dtype();
//     CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
//     auto result = PixelShuffleInfo::createPixelShuffleInfo(
//         output_desc,
//         input_desc,
//         upscale_factor);
//     CHECK_RESULT(result);
//     const PixelShuffleInfo &info = result.take();

//     // Debug output for all dtypes when non-contiguous
//     bool is_non_contiguous = (info.input_w_stride != 1) || 
//                              (info.input_h_stride != info.input_w_stride * (info.H_out / info.r)) ||
//                              (info.input_c_stride != info.input_h_stride * (info.H_out / info.r) * (info.W_out / info.r));
    
//     // No workspace needed for the new kernel
//     size_t WorkSpaceSize = 0;
    
//     *desc_ptr = new Descriptor(
//         dtype, std::move(info), WorkSpaceSize,
//         new Opaque{handle->internal()},
//         handle->device, handle->device_id);
//     return INFINI_STATUS_SUCCESS;
// }

// infiniStatus_t Descriptor::calculate(
//     void *workspace,
//     size_t workspace_size,
//     void *output,
//     const void *input,
//     void *stream_) const {
    
//     if (workspace_size < _workspace_size) {
//         return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
//     }
//     cudaStream_t stream = (cudaStream_t)stream_;

// #define CALCULATE_PIXEL_SHUFFLE(TDATA) 
//     calculate_pixel_shuffle<TDATA>(_info, (TDATA *)output, (const TDATA *)input, stream, workspace)

//     if (_info.dtype == INFINI_DTYPE_F16) {
//         return CALCULATE_PIXEL_SHUFFLE(half);
//     } else if (_info.dtype == INFINI_DTYPE_F32) {
//         return CALCULATE_PIXEL_SHUFFLE(float);
//     } else if (_info.dtype == INFINI_DTYPE_BF16) {
//         return CALCULATE_PIXEL_SHUFFLE(__nv_bfloat16);
//     } else {
//         return INFINI_STATUS_BAD_TENSOR_DTYPE;
//     }

//     return INFINI_STATUS_SUCCESS;
// }

// } // namespace op::pixel_shuffle::nvidia


#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cstdio>

#include "../cuda/kernel.cuh"
#include "../info.h"
#include "pixel_shuffle_nvidia.cuh"

namespace op::pixel_shuffle::nvidia {

// 添加 Opaque 结构体定义
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

// 添加析构函数实现
Descriptor::~Descriptor() {
    delete _opaque;
}

// 添加 create 函数实现
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int upscale_factor) {
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    
    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
    auto result = PixelShuffleInfo::createPixelShuffleInfo(
        output_desc,
        input_desc,
        upscale_factor);
    CHECK_RESULT(result);
    const PixelShuffleInfo &info = result.take();
    
    // Debug: Print stride info (always print for diagnosis)
    // 计算输入维度（用于检测连续张量）
    int input_w = info.W_out / info.r;
    int input_h = info.H_out / info.r;
    int input_c = info.C_out * info.r * info.r;
    
    // 连续输入张量的期望 stride
    int expected_w_stride = 1;
    int expected_h_stride = input_w;
    int expected_c_stride = input_h * input_w;
    int expected_b_stride = input_c * input_h * input_w;
    
    bool is_non_contiguous = (info.input_w_stride != expected_w_stride) || 
                             (info.input_h_stride != expected_h_stride) ||
                             (info.input_c_stride != expected_c_stride) ||
                             (info.input_b_stride != expected_b_stride);
    
    // Always print debug info
    int dtype_val = static_cast<int>(info.dtype);
    int r_val = static_cast<int>(info.r);
    printf("DEBUG [PIXEL_SHUFFLE]: DType=%d, Shape=(%d,%d,%d,%d), r=%d\n",
           dtype_val, info.B, info.C_out, info.H_out, info.W_out, r_val);
    printf("DEBUG [PIXEL_SHUFFLE]: Input dims: C=%d, H=%d, W=%d\n",
           input_c, input_h, input_w);
    printf("DEBUG [PIXEL_SHUFFLE]: Input strides: B=%ld, C=%ld, H=%ld, W=%ld\n",
           (long)info.input_b_stride, (long)info.input_c_stride, (long)info.input_h_stride, (long)info.input_w_stride);
    printf("DEBUG [PIXEL_SHUFFLE]: Output strides: B=%ld, C=%ld, H=%ld, W=%ld\n",
           (long)info.output_b_stride, (long)info.output_c_stride, (long)info.output_h_stride, (long)info.output_w_stride);
    printf("DEBUG [PIXEL_SHUFFLE]: Expected input strides: B=%d, C=%d, H=%d, W=%d\n",
           expected_b_stride, expected_c_stride, expected_h_stride, expected_w_stride);
    printf("DEBUG [PIXEL_SHUFFLE]: Is non-contiguous: %d\n", is_non_contiguous ? 1 : 0);
    fflush(stdout);
    
    // No workspace needed for the new kernel
    size_t WorkSpaceSize = 0;
    
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// 然后才是 calculate_pixel_shuffle 模板函数
template <typename T>
infiniStatus_t calculate_pixel_shuffle(
    const PixelShuffleInfo& info,
    T* output,
    const T* input,
    cudaStream_t stream,
    void* workspace)
{
    (void)workspace;

    dim3 block(16, 16);
    dim3 grid(
        (info.W_out + block.x - 1) / block.x,
        (info.H_out + block.y - 1) / block.y,
        info.B * info.C_out
    );

    pixel_shuffle_kernel<T><<<grid, block, 0, stream>>>(
        input,
        output,
        info.r,
        info.B,
        info.C_out,
        info.H_out,
        info.W_out,
        static_cast<size_t>(info.input_b_stride),
        static_cast<size_t>(info.input_c_stride),
        static_cast<size_t>(info.input_h_stride),
        static_cast<size_t>(info.input_w_stride),
        static_cast<size_t>(info.output_b_stride),
        static_cast<size_t>(info.output_c_stride),
        static_cast<size_t>(info.output_h_stride),
        static_cast<size_t>(info.output_w_stride)
    );

    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void* workspace,
    size_t workspace_size,
    void* output,
    const void* input,
    void* stream_) const
{
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    cudaStream_t stream = (cudaStream_t)stream_;

#define DISPATCH(T) \
    return calculate_pixel_shuffle<T>(_info, (T*)output, (const T*)input, stream, workspace)

    switch (_info.dtype) {
        case INFINI_DTYPE_F16:  DISPATCH(half);
        case INFINI_DTYPE_F32:  DISPATCH(float);
        case INFINI_DTYPE_BF16: DISPATCH(__nv_bfloat16);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#undef DISPATCH
}

} // namespace op::pixel_shuffle::nvidia
