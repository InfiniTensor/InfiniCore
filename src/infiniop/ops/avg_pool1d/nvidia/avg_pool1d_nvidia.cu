#include "../../../devices/nvidia/nvidia_common.cuh"
#include "avg_pool1d_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"

// 1. 定义 Global Kernel 入口
// 这层包装是为了把 device 函数暴露给 host 调用
template <typename T>
__global__ void avgPool1dGlobalKernel(
    T *y,
    const T *x,
    size_t batch,
    size_t channels,
    size_t in_width,
    size_t out_width,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    ptrdiff_t y_stride_batch,
    ptrdiff_t y_stride_channel,
    ptrdiff_t y_stride_width,
    ptrdiff_t x_stride_batch,
    ptrdiff_t x_stride_channel,
    ptrdiff_t x_stride_width) {
    
    // 调用 kernel.cuh 中的 device 逻辑
    avgPool1dKernel<T>(
        y, x, 
        batch, channels, in_width, out_width, 
        kernel_size, stride, padding,
        y_stride_batch, y_stride_channel, y_stride_width,
        x_stride_batch, x_stride_channel, x_stride_width
    );
}

namespace op::avg_pool1d::nvidia {

// 2. 定义 Opaque 结构 (持有 NVIDIA Handle)
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

// 3. 实现 Create 函数
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t kernel_size,
    size_t stride,
    size_t padding) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    // 使用 Info 类进行参数校验和预计算
    auto info = AvgPool1dInfo::createAvgPool1dInfo(y_desc, x_desc, kernel_size, stride, padding);
    CHECK_RESULT(info);

    // 创建 Descriptor
    *desc_ptr = new Descriptor(
        info.take(),
        0, // workspace size
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// 4. 实现计算辅助函数
template <typename T>
infiniStatus_t calculateAvgPool1d(
    const AvgPool1dInfo &info,
    int max_threads_per_block,
    T *y,
    const T *x,
    cudaStream_t stream) {

    // 计算总任务量: Batch * Channel * OutWidth 
    size_t total_elements = info.batch * info.channels * info.out_width;
    
    // 确定 Block 和 Grid 大小
    int block_size = 256; 
    if (max_threads_per_block > 0 && max_threads_per_block < 256) {
        block_size = max_threads_per_block;
    }
    
    // 简单的 1D Grid 策略，配合 kernel.cuh 里的 Grid-Stride Loop
    // 限制 grid 大小以防过大，通常 65535 或根据 SM 数量调整即可
    size_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; 

    avgPool1dGlobalKernel<T><<<grid_size, block_size, 0, stream>>>(
        y, x,
        info.batch, info.channels, info.in_width, info.out_width,
        info.kernel_size, info.stride, info.padding,
        info.y_stride_batch, info.y_stride_channel, info.y_stride_width,
        info.x_stride_batch, info.x_stride_channel, info.x_stride_width
    );

    return INFINI_STATUS_SUCCESS;
}

// 5. 宏定义与类型分发
#define CALCULATE(TDATA) \
    calculateAvgPool1d(_info, \
                       _opaque->internal->maxThreadsPerBlock(), \
                       (TDATA *)y, \
                       (const TDATA *)x, \
                       (cudaStream_t)stream)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return CALCULATE(half);
    case INFINI_DTYPE_BF16:
        return CALCULATE(cuda_bfloat16);
    case INFINI_DTYPE_F32:
        return CALCULATE(float);
    case INFINI_DTYPE_F64:
        return CALCULATE(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE

} // namespace op::avg_pool1d::nvidia