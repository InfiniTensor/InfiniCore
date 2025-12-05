#ifndef __REARRANGE_CUDA_KERNEL_H__
#define __REARRANGE_CUDA_KERNEL_H__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

// 简化的kernel，直接使用RearrangeMeta的布局
template <typename T>
__global__ void rearrange_kernel_simple(
    void *__restrict__ dst,
    const void *__restrict__ src,
    const size_t count,
    const size_t unit,
    const size_t ndim,
    const ptrdiff_t *idx_strides,
    const ptrdiff_t *dst_strides,
    const ptrdiff_t *src_strides) {

    // 每个线程处理一个unit
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count) {
        return;
    }

    // 计算在dst和src中的偏移
    ptrdiff_t dst_offset = 0;
    ptrdiff_t src_offset = 0;
    size_t rem = idx;

    for (size_t i = 0; i < ndim; ++i) {
        size_t k = rem / idx_strides[i];
        dst_offset += k * dst_strides[i];
        src_offset += k * src_strides[i];
        rem %= idx_strides[i];
    }

    // 执行拷贝
    char *dst_ptr = reinterpret_cast<char *>(dst) + dst_offset;
    const char *src_ptr = reinterpret_cast<const char *>(src) + src_offset;

    // 根据unit大小进行拷贝
    if (sizeof(T) == unit) {
        *reinterpret_cast<T *>(dst_ptr) = *reinterpret_cast<const T *>(src_ptr);
    } else {
        // 逐字节拷贝
        for (size_t i = 0; i < unit; ++i) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

// 简化的参数结构
struct RearrangeParams {
    size_t count;
    size_t unit;
    size_t ndim;
    std::vector<ptrdiff_t> idx_strides;
    std::vector<ptrdiff_t> dst_strides;
    std::vector<ptrdiff_t> src_strides;
    size_t unit_size; // 原始unit大小，用于选择kernel
};

// 获取kernel函数指针
utils::Result<void *> getRearrangeKernel(const RearrangeParams &params) {
    // 根据unit大小选择kernel
    void *kernel_func = nullptr;

    switch (params.unit_size) {
    case 1:
        kernel_func = (void *)rearrange_kernel_simple<uint8_t>;
        break;
    case 2:
        kernel_func = (void *)rearrange_kernel_simple<uint16_t>;
        break;
    case 4:
        kernel_func = (void *)rearrange_kernel_simple<uint32_t>;
        break;
    case 8:
        kernel_func = (void *)rearrange_kernel_simple<uint64_t>;
        break;
    case 16:
        kernel_func = (void *)rearrange_kernel_simple<float4>;
        break;
    case 32:
        kernel_func = (void *)rearrange_kernel_simple<double4>;
        break;
    default:
        // 对于不支持的unit大小，使用字节拷贝kernel
        kernel_func = (void *)rearrange_kernel_simple<uint8_t>;
        break;
    }

    return utils::Result<void *>(kernel_func);
}

#endif // __REARRANGE_CUDA_KERNEL_H__
