#include "../../../devices/nvidia/nvidia_common.cuh"
#include "cross_entropy_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh" // 引入刚才写的 kernel

// ----------------------------------------------------------------------
// Wrapper: 包装 kernel 调用，方便 launchKernel 使用
// ----------------------------------------------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tidx, typename Tcompute = float>
INFINIOP_CUDA_KERNEL crossEntropy(
    Tdata *y, const Tdata *x, const void *target,
    size_t outer_size, size_t vocab_size, ptrdiff_t x_stride) {
    
    // 调用 device 函数
    crossEntropyKernel<BLOCK_SIZE, Tdata, Tidx, Tcompute>(
        y, x, target, outer_size, vocab_size, x_stride
    );
}

namespace op::cross_entropy::nvidia {

// ----------------------------------------------------------------------
// Opaque 结构体: 存储 NVIDIA Handle (用于获取 device 属性)
// ----------------------------------------------------------------------
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

// ----------------------------------------------------------------------
// Create: 初始化 Info 并创建 Descriptor
// ----------------------------------------------------------------------
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t target_desc) {
    
    // 1. 基础校验 (复用 CPU 逻辑或重写)
    auto x_dtype = x_desc->dtype();
    auto t_dtype = target_desc->dtype();
    // CHECK_DTYPE(x_dtype, ...); // 如果有公用宏
    
    // 2. 填充 Info
    CrossEntropyInfo info;
    info.dtype = x_dtype;
    info.target_dtype = t_dtype;
    
    info.vocab_size = x_desc->shape().back();
    info.outer_size = target_desc->numel(); // Batch * Seq
    info.x_stride = static_cast<ptrdiff_t>(info.vocab_size); // 假设连续

    // 3. 创建 Opaque
    auto internal = reinterpret_cast<device::nvidia::Handle *>(handle)->internal();
    
    *desc_ptr = new Descriptor(
        new Opaque{internal},
        info, 0, handle->device, handle->device_id
    );
    return INFINI_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------
// Launch Kernel: 负责 Grid 计算和 Template 实例化
// ----------------------------------------------------------------------
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, const void *target, 
                            const CrossEntropyInfo &info, cudaStream_t stream) {
    
    // Grid 策略: 
    // blockIdx.x 对应 Outer 维度 (每一行一个 Block)
    // 这种策略对于 Vocab 非常大的情况 (如 32k, 128k) 是合理的
    // 如果 Outer 很大 (>65535)，需要注意 gridDim.x 的限制 (通常 max 2^31-1，够用)
    dim3 grid(static_cast<uint32_t>(info.outer_size), 1, 1);
    
    // 双重分发: Logits Dtype * Target Dtype
    if (info.target_dtype == INFINI_DTYPE_I64) {
        if (info.dtype == INFINI_DTYPE_F16) {
             crossEntropy<BLOCK_SIZE, half, int64_t>
                 <<<grid, BLOCK_SIZE, 0, stream>>>((half*)y, (const half*)x, target, info.outer_size, info.vocab_size, info.x_stride);
        } else if (info.dtype == INFINI_DTYPE_BF16) {
             crossEntropy<BLOCK_SIZE, __nv_bfloat16, int64_t>
                 <<<grid, BLOCK_SIZE, 0, stream>>>((__nv_bfloat16*)y, (const __nv_bfloat16*)x, target, info.outer_size, info.vocab_size, info.x_stride);
        } else if (info.dtype == INFINI_DTYPE_F32) {
             crossEntropy<BLOCK_SIZE, float, int64_t>
                 <<<grid, BLOCK_SIZE, 0, stream>>>((float*)y, (const float*)x, target, info.outer_size, info.vocab_size, info.x_stride);
        }
    } else if (info.target_dtype == INFINI_DTYPE_I32) {
        // 类似的逻辑，针对 int32 target
        if (info.dtype == INFINI_DTYPE_F16) {
             crossEntropy<BLOCK_SIZE, half, int32_t>
                 <<<grid, BLOCK_SIZE, 0, stream>>>((half*)y, (const half*)x, target, info.outer_size, info.vocab_size, info.x_stride);
        } else if (info.dtype == INFINI_DTYPE_BF16) {
             crossEntropy<BLOCK_SIZE, __nv_bfloat16, int32_t>
                 <<<grid, BLOCK_SIZE, 0, stream>>>((__nv_bfloat16*)y, (const __nv_bfloat16*)x, target, info.outer_size, info.vocab_size, info.x_stride);
        } else if (info.dtype == INFINI_DTYPE_F32) {
             crossEntropy<BLOCK_SIZE, float, int32_t>
                 <<<grid, BLOCK_SIZE, 0, stream>>>((float*)y, (const float*)x, target, info.outer_size, info.vocab_size, info.x_stride);
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------
// Calculate: 选择 Block Size 并执行
// ----------------------------------------------------------------------
infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     const void *target,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    
    // 根据 GPU 架构或 Vocab Size 选择合适的 Block Size
    // 这里简单地根据设备支持的最大 Block Size 分发
    int max_threads = _opaque->internal->maxThreadsPerBlock();
    
    // 对于 Reduction Kernel，Block Size 越大通常越好 (减少 Block 数量或增加并行度)
    // 但不能超过 Vocab Size 太多
    
    if (max_threads >= 1024) {
        CHECK_STATUS(launchKernel<1024>(y, x, target, _info, stream));
    } else if (max_threads >= 512) {
        CHECK_STATUS(launchKernel<512>(y, x, target, _info, stream));
    } else {
        CHECK_STATUS(launchKernel<256>(y, x, target, _info, stream));
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::cross_entropy::nvidia