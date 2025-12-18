#ifndef __REARRANGE_TRANSPOSE_KERNEL_CUH__
#define __REARRANGE_TRANSPOSE_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

namespace op::rearrange::nvidia {

// Shared memory tile大小配置
// 使用32x32 tile以获得最佳bank conflict避免和性能
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;  // 每个线程处理多行以提高occupancy

/**
 * 转置模式信息结构
 */
struct TransposeInfo {
    size_t ndim;              // 维度数
    size_t total_elements;    // 总元素数
    size_t element_size;      // 元素大小（字节）
    
    // 将多维索引展平的参数
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> src_strides;  // 源stride（字节）
    std::vector<ptrdiff_t> dst_strides;  // 目标stride（字节）
};

/**
 * 2D转置kernel - 使用shared memory tiling
 * 适用于简单的2D矩阵转置
 */
template <typename T>
__global__ void transpose_2d_kernel_tiled(
    T *__restrict__ dst,
    const T *__restrict__ src,
    size_t rows,
    size_t cols,
    ptrdiff_t src_row_stride,  // 以T为单位
    ptrdiff_t src_col_stride,
    ptrdiff_t dst_row_stride,
    ptrdiff_t dst_col_stride) {
    
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];  // +1 避免bank conflict
    
    // 计算全局坐标
    size_t x = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 从源读取到shared memory（coalesced读取）
    if (x < cols && y < rows) {
        size_t src_idx = y * src_row_stride + x * src_col_stride;
        tile[threadIdx.y][threadIdx.x] = src[src_idx];
    }
    
    __syncthreads();
    
    // 转置后的坐标
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // 从shared memory写入到目标（coalesced写入）
    if (x < rows && y < cols) {
        size_t dst_idx = y * dst_row_stride + x * dst_col_stride;
        dst[dst_idx] = tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * 多维转置kernel - 通用版本
 * 处理任意维度的行列主序转换
 * 
 * 策略：使用shared memory作为缓冲区，分批处理
 */
template <typename T, int TILE_SIZE = 32>
__global__ void transpose_nd_kernel(
    T *__restrict__ dst,
    const T *__restrict__ src,
    const size_t *shape,         // [ndim]
    const ptrdiff_t *src_strides, // [ndim] 以T为单位
    const ptrdiff_t *dst_strides, // [ndim] 以T为单位
    size_t ndim,
    size_t total_elements) {
    
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= total_elements) {
        return;
    }
    
    // 计算多维索引
    size_t remaining = gid;
    size_t src_offset = 0;
    size_t dst_offset = 0;
    
    // 使用局部数组存储中间结果以减少寄存器压力
    size_t indices[8];  // 支持最多8维
    
    // 从线性索引计算多维索引
    for (int i = ndim - 1; i >= 0; i--) {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    
    // 计算源和目标偏移
    for (size_t i = 0; i < ndim; i++) {
        src_offset += indices[i] * src_strides[i];
        dst_offset += indices[i] * dst_strides[i];
    }
    
    // 执行拷贝
    dst[dst_offset] = src[src_offset];
}

/**
 * 向量化的多维转置kernel
 * 当数据对齐时使用4元素向量化访问
 */
template <typename T>
__global__ void transpose_nd_kernel_vec4(
    T *__restrict__ dst,
    const T *__restrict__ src,
    const size_t *shape,
    const ptrdiff_t *src_strides,
    const ptrdiff_t *dst_strides,
    size_t ndim,
    size_t total_elements) {
    
    size_t gid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (gid + 3 >= total_elements) {
        // 处理边界情况
        for (size_t i = gid; i < total_elements && i < gid + 4; i++) {
            size_t remaining = i;
            size_t src_offset = 0;
            size_t dst_offset = 0;
            
            size_t indices[8];
            for (int j = ndim - 1; j >= 0; j--) {
                indices[j] = remaining % shape[j];
                remaining /= shape[j];
            }
            
            for (size_t j = 0; j < ndim; j++) {
                src_offset += indices[j] * src_strides[j];
                dst_offset += indices[j] * dst_strides[j];
            }
            
            dst[dst_offset] = src[src_offset];
        }
        return;
    }
    
    // 向量化处理4个元素
    for (int k = 0; k < 4; k++) {
        size_t i = gid + k;
        size_t remaining = i;
        size_t src_offset = 0;
        size_t dst_offset = 0;
        
        size_t indices[8];
        for (int j = ndim - 1; j >= 0; j--) {
            indices[j] = remaining % shape[j];
            remaining /= shape[j];
        }
        
        for (size_t j = 0; j < ndim; j++) {
            src_offset += indices[j] * src_strides[j];
            dst_offset += indices[j] * dst_strides[j];
        }
        
        dst[dst_offset] = src[src_offset];
    }
}

/**
 * 6D特化的转置kernel - 针对(3,4,50,50,5,7)这类case优化
 * 使用更好的索引计算和cache策略
 */
template <typename T>
__global__ void transpose_6d_kernel_optimized(
    T *__restrict__ dst,
    const T *__restrict__ src,
    size_t d0, size_t d1, size_t d2, size_t d3, size_t d4, size_t d5,
    ptrdiff_t s0, ptrdiff_t s1, ptrdiff_t s2, ptrdiff_t s3, ptrdiff_t s4, ptrdiff_t s5,
    ptrdiff_t d_s0, ptrdiff_t d_s1, ptrdiff_t d_s2, ptrdiff_t d_s3, ptrdiff_t d_s4, ptrdiff_t d_s5,
    size_t total_elements) {
    
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= total_elements) {
        return;
    }
    
    // 手动展开索引计算以减少除法操作
    size_t idx = gid;
    size_t i5 = idx % d5; idx /= d5;
    size_t i4 = idx % d4; idx /= d4;
    size_t i3 = idx % d3; idx /= d3;
    size_t i2 = idx % d2; idx /= d2;
    size_t i1 = idx % d1; idx /= d1;
    size_t i0 = idx;
    
    // 计算源和目标偏移
    size_t src_offset = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4 + i5 * s5;
    size_t dst_offset = i0 * d_s0 + i1 * d_s1 + i2 * d_s2 + i3 * d_s3 + i4 * d_s4 + i5 * d_s5;
    
    dst[dst_offset] = src[src_offset];
}

} // namespace op::rearrange::nvidia

#endif // __REARRANGE_TRANSPOSE_KERNEL_CUH__

