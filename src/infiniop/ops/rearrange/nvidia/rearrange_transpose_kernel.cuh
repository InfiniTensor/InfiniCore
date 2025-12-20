#ifndef __REARRANGE_TRANSPOSE_KERNEL_CUH__
#define __REARRANGE_TRANSPOSE_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

namespace op::rearrange::nvidia {

// Shared memory tile大小配置
// 使用32x32 tile以获得最佳bank conflict避免和性能
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;  // 每个线程处理多行以提高occupancy

// 小矩阵专用 tile 配置：降低 launch/同步开销
constexpr int TILE_DIM_SMALL = 16;
constexpr int BLOCK_ROWS_SMALL = 8;

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
    // 采用 BLOCK_ROWS 提升 occupancy：每个线程负责多行
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t yj = y + j;
        if (x < cols && yj < rows) {
            size_t src_idx = yj * src_row_stride + x * src_col_stride;
            tile[threadIdx.y + j][threadIdx.x] = src[src_idx];
        }
    }
    
    __syncthreads();
    
    // 转置后的坐标
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // 从shared memory写入到目标（coalesced写入）
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t yj = y + j;
        if (x < rows && yj < cols) {
            size_t dst_idx = yj * dst_row_stride + x * dst_col_stride;
            dst[dst_idx] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

/**
 * 2D 转置 kernel（小矩阵版本，16x16 tile）
 *
 * 对于 100x100 这类矩阵，32x32 tile 的 block/smem/sync 开销占比更高；
 * 16x16 tile 往往能更接近 PyTorch 的小矩阵性能。
 */
template <typename T>
__global__ void transpose_2d_kernel_tiled_small(
    T *__restrict__ dst,
    const T *__restrict__ src,
    size_t rows,
    size_t cols,
    ptrdiff_t src_row_stride,  // 以T为单位
    ptrdiff_t src_col_stride,
    ptrdiff_t dst_row_stride,
    ptrdiff_t dst_col_stride) {

    __shared__ T tile[TILE_DIM_SMALL][TILE_DIM_SMALL + 1];

    size_t x = blockIdx.x * TILE_DIM_SMALL + threadIdx.x;
    size_t y = blockIdx.y * TILE_DIM_SMALL + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM_SMALL; j += BLOCK_ROWS_SMALL) {
        size_t yj = y + j;
        if (x < cols && yj < rows) {
            size_t src_idx = yj * src_row_stride + x * src_col_stride;
            tile[threadIdx.y + j][threadIdx.x] = src[src_idx];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM_SMALL + threadIdx.x;
    y = blockIdx.x * TILE_DIM_SMALL + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM_SMALL; j += BLOCK_ROWS_SMALL) {
        size_t yj = y + j;
        if (x < rows && yj < cols) {
            size_t dst_idx = yj * dst_row_stride + x * dst_col_stride;
            dst[dst_idx] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

/**
 * 2D 转置 kernel（unit==2 专用，2元素向量化）
 *
 * block.x=16，每线程处理两个相邻列（共覆盖 32 列），shared memory 仍为 32x(32+1)。
 * 适用于 cols 为偶数且 src/dst 在列方向 stride=1 的典型 transpose（如 row<->col major）。
 */
__global__ void transpose_2d_kernel_tiled_u16x2(
    uint16_t *__restrict__ dst,
    const uint16_t *__restrict__ src,
    size_t rows,
    size_t cols,
    ptrdiff_t src_row_stride,  // 以 uint16_t 为单位
    ptrdiff_t src_col_stride,
    ptrdiff_t dst_row_stride,
    ptrdiff_t dst_col_stride) {

    __shared__ uint16_t tile[TILE_DIM][TILE_DIM + 1];

    // 每线程覆盖 2 列
    size_t x = blockIdx.x * TILE_DIM + threadIdx.x * 2;
    size_t y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load: src -> tile
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t yj = y + j;
        if (yj < rows) {
            size_t x0 = x;
            size_t x1 = x + 1;
            if (x0 < cols) {
                size_t src_idx0 = yj * src_row_stride + x0 * src_col_stride;
                tile[threadIdx.y + j][threadIdx.x * 2] = src[src_idx0];
            }
            if (x1 < cols) {
                size_t src_idx1 = yj * src_row_stride + x1 * src_col_stride;
                tile[threadIdx.y + j][threadIdx.x * 2 + 1] = src[src_idx1];
            }
        }
    }

    __syncthreads();

    // Store: tile^T -> dst
    x = blockIdx.y * TILE_DIM + threadIdx.x * 2;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t yj = y + j;
        if (yj < cols) {
            size_t x0 = x;
            size_t x1 = x + 1;
            if (x0 < rows) {
                size_t dst_idx0 = yj * dst_row_stride + x0 * dst_col_stride;
                dst[dst_idx0] = tile[threadIdx.x * 2][threadIdx.y + j];
            }
            if (x1 < rows) {
                size_t dst_idx1 = yj * dst_row_stride + x1 * dst_col_stride;
                dst[dst_idx1] = tile[threadIdx.x * 2 + 1][threadIdx.y + j];
            }
        }
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

/**
 * 6D 转置 kernel（增量进位版本）
 *
 * 关键点：每个线程处理 VEC 个连续的 gid。
 * 只对第一个 gid 做 div/mod，后续用 +stride 和进位修正 offset，降低索引开销。
 */
template <typename T, int VEC = 4>
__global__ void transpose_6d_kernel_inc(
    T *__restrict__ dst,
    const T *__restrict__ src,
    size_t d0, size_t d1, size_t d2, size_t d3, size_t d4, size_t d5,
    ptrdiff_t s0, ptrdiff_t s1, ptrdiff_t s2, ptrdiff_t s3, ptrdiff_t s4, ptrdiff_t s5,
    ptrdiff_t t0, ptrdiff_t t1, ptrdiff_t t2, ptrdiff_t t3, ptrdiff_t t4, ptrdiff_t t5,
    size_t total_elements) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gid0 = tid * static_cast<size_t>(VEC);
    if (gid0 >= total_elements) return;

    // 先把 gid0 解成 6D index（这一步仍然需要 div/mod，但每线程只做一次）
    size_t idx = gid0;
    size_t i5 = idx % d5; idx /= d5;
    size_t i4 = idx % d4; idx /= d4;
    size_t i3 = idx % d3; idx /= d3;
    size_t i2 = idx % d2; idx /= d2;
    size_t i1 = idx % d1; idx /= d1;
    size_t i0 = idx;

    ptrdiff_t src_offset = static_cast<ptrdiff_t>(i0) * s0 +
                           static_cast<ptrdiff_t>(i1) * s1 +
                           static_cast<ptrdiff_t>(i2) * s2 +
                           static_cast<ptrdiff_t>(i3) * s3 +
                           static_cast<ptrdiff_t>(i4) * s4 +
                           static_cast<ptrdiff_t>(i5) * s5;

    ptrdiff_t dst_offset = static_cast<ptrdiff_t>(i0) * t0 +
                           static_cast<ptrdiff_t>(i1) * t1 +
                           static_cast<ptrdiff_t>(i2) * t2 +
                           static_cast<ptrdiff_t>(i3) * t3 +
                           static_cast<ptrdiff_t>(i4) * t4 +
                           static_cast<ptrdiff_t>(i5) * t5;

    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
        size_t gid = gid0 + static_cast<size_t>(k);
        if (gid >= total_elements) break;

        dst[dst_offset] = src[src_offset];

        // i5++ 以及进位（同时修正 src/dst offset）
        i5 += 1;
        src_offset += s5;
        dst_offset += t5;

        if (i5 == d5) {
            i5 = 0;
            src_offset += s4 - static_cast<ptrdiff_t>(d5) * s5;
            dst_offset += t4 - static_cast<ptrdiff_t>(d5) * t5;
            i4 += 1;

            if (i4 == d4) {
                i4 = 0;
                src_offset += s3 - static_cast<ptrdiff_t>(d4) * s4;
                dst_offset += t3 - static_cast<ptrdiff_t>(d4) * t4;
                i3 += 1;

                if (i3 == d3) {
                    i3 = 0;
                    src_offset += s2 - static_cast<ptrdiff_t>(d3) * s3;
                    dst_offset += t2 - static_cast<ptrdiff_t>(d3) * t3;
                    i2 += 1;

                    if (i2 == d2) {
                        i2 = 0;
                        src_offset += s1 - static_cast<ptrdiff_t>(d2) * s2;
                        dst_offset += t1 - static_cast<ptrdiff_t>(d2) * t2;
                        i1 += 1;

                        if (i1 == d1) {
                            i1 = 0;
                            src_offset += s0 - static_cast<ptrdiff_t>(d1) * s1;
                            dst_offset += t0 - static_cast<ptrdiff_t>(d1) * t1;
                            i0 += 1;
                        }
                    }
                }
            }
        }
    }
}

/**
 * 5D 转置 kernel（增量进位版本）
 *
 * 每线程处理 VEC 个连续 gid，只对第一个 gid 做 div/mod；
 * 后续通过 +stride + 进位修正 offset，减少索引开销。
 */
template <typename T, int VEC = 4>
__global__ void transpose_5d_kernel_inc(
    T *__restrict__ dst,
    const T *__restrict__ src,
    size_t d0, size_t d1, size_t d2, size_t d3, size_t d4,
    ptrdiff_t s0, ptrdiff_t s1, ptrdiff_t s2, ptrdiff_t s3, ptrdiff_t s4,
    ptrdiff_t t0, ptrdiff_t t1, ptrdiff_t t2, ptrdiff_t t3, ptrdiff_t t4,
    size_t total_elements) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gid0 = tid * static_cast<size_t>(VEC);
    if (gid0 >= total_elements) return;

    size_t idx = gid0;
    size_t i4 = idx % d4; idx /= d4;
    size_t i3 = idx % d3; idx /= d3;
    size_t i2 = idx % d2; idx /= d2;
    size_t i1 = idx % d1; idx /= d1;
    size_t i0 = idx;

    ptrdiff_t src_offset = static_cast<ptrdiff_t>(i0) * s0 +
                           static_cast<ptrdiff_t>(i1) * s1 +
                           static_cast<ptrdiff_t>(i2) * s2 +
                           static_cast<ptrdiff_t>(i3) * s3 +
                           static_cast<ptrdiff_t>(i4) * s4;

    ptrdiff_t dst_offset = static_cast<ptrdiff_t>(i0) * t0 +
                           static_cast<ptrdiff_t>(i1) * t1 +
                           static_cast<ptrdiff_t>(i2) * t2 +
                           static_cast<ptrdiff_t>(i3) * t3 +
                           static_cast<ptrdiff_t>(i4) * t4;

    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
        size_t gid = gid0 + static_cast<size_t>(k);
        if (gid >= total_elements) break;

        dst[dst_offset] = src[src_offset];

        // i4++ and carry
        i4 += 1;
        src_offset += s4;
        dst_offset += t4;

        if (i4 == d4) {
            i4 = 0;
            src_offset += s3 - static_cast<ptrdiff_t>(d4) * s4;
            dst_offset += t3 - static_cast<ptrdiff_t>(d4) * t4;
            i3 += 1;

            if (i3 == d3) {
                i3 = 0;
                src_offset += s2 - static_cast<ptrdiff_t>(d3) * s3;
                dst_offset += t2 - static_cast<ptrdiff_t>(d3) * t3;
                i2 += 1;

                if (i2 == d2) {
                    i2 = 0;
                    src_offset += s1 - static_cast<ptrdiff_t>(d2) * s2;
                    dst_offset += t1 - static_cast<ptrdiff_t>(d2) * t2;
                    i1 += 1;

                    if (i1 == d1) {
                        i1 = 0;
                        src_offset += s0 - static_cast<ptrdiff_t>(d1) * s1;
                        dst_offset += t0 - static_cast<ptrdiff_t>(d1) * t1;
                        i0 += 1;
                    }
                }
            }
        }
    }
}

} // namespace op::rearrange::nvidia

#endif // __REARRANGE_TRANSPOSE_KERNEL_CUH__

