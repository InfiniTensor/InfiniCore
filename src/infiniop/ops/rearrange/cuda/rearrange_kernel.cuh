#ifndef __REARRANGE_CUDA_KERNEL_H__
#define __REARRANGE_CUDA_KERNEL_H__

#include "../../../devices/cuda/cuda_common.cuh"

#define ARRAY_TYPE int

template <int ArrSize, typename ArrayType>
struct ArrayStruct {
    ArrayType a[ArrSize];
};

// 各个元素分别代表：[grid_idx, block_idx, grid的stride相对于block的倍数，总的len限制]
template <typename ElementType>
struct Constrains {
    ElementType grid_idx;
    ElementType block_idx;
    ElementType grid_div_block;
    ElementType total_len;
};

#define IF_CONSTRAIN_0 , const ArrayStruct<1, Constrains<ARRAY_TYPE>> constrains
#define IF_CONSTRAIN_1 , const ArrayStruct<1, Constrains<ARRAY_TYPE>> constrains
#define IF_CONSTRAIN_2 , const ArrayStruct<2, Constrains<ARRAY_TYPE>> constrains

// 定义宏生成内核函数
#define DEFINE_REARRANGE_KERNEL(Tmem_type, constrain_num, block_array_size, grid_array_size)                                                                                     \
    extern "C" __global__ void rearrange_unit_##Tmem_type##_block_##block_array_size##_grid_##grid_array_size##_constrain_##constrain_num(                                       \
        void *__restrict__ dst,                                                                                                                                                  \
        void const *__restrict__ src,                                                                                                                                            \
        unsigned int const block_dim,                                                                                                                                            \
        unsigned int const block_len_total,                                                                                                                                      \
        const ArrayStruct<block_array_size, ARRAY_TYPE> block_len,                                                                                                               \
        const ArrayStruct<block_array_size, ARRAY_TYPE> src_block_stride, /* 字节单位的步长 */                                                                            \
        const ArrayStruct<block_array_size, ARRAY_TYPE> dst_block_stride, /* 字节单位的步长 */                                                                            \
        const ArrayStruct<grid_array_size, ARRAY_TYPE> grid_len,                                                                                                                 \
        const ArrayStruct<grid_array_size, ARRAY_TYPE> src_grid_stride, /* 字节单位的步长 */                                                                              \
        const ArrayStruct<grid_array_size, ARRAY_TYPE> dst_grid_stride  /* 字节单位的步长 */                                                                              \
            IF_CONSTRAIN_##constrain_num) {                                                                                                                                      \
        int remaining = threadIdx.x;                                                                                                                                             \
        if (remaining >= block_len_total) {                                                                                                                                      \
            return;                                                                                                                                                              \
        }                                                                                                                                                                        \
                                                                                                                                                                                 \
        /* 声明共享内存 */                                                                                                                                                 \
        __shared__ int shared_src_offset;                                                                                                                                        \
        __shared__ int shared_dst_offset;                                                                                                                                        \
                                                                                                                                                                                 \
        if (constrain_num > 0) {                                                                                                                                                 \
            __shared__ int shared_constrains_grid_idx_multiple[constrain_num > 0 ? constrain_num : 1];                                                                           \
                                                                                                                                                                                 \
            if (threadIdx.x == 0) { /* 只让0号线程计算 */                                                                                                                 \
                /* 计算当前block处理的数据在src和dst中的基础偏移(bytes) */                                                                                      \
                int src_offset = 0;                                                                                                                                              \
                int dst_offset = 0;                                                                                                                                              \
                int constrains_grid_idx_multiple[constrain_num > 0 ? constrain_num : 1];                                                                                         \
                                                                                                                                                                                 \
                int remaining                                                                                                                                                    \
                    = blockIdx.x;                                                                                                                                                \
                                                                                                                                                                                 \
                for (int i = grid_array_size - 1; i >= 0; i--) {                                                                                                                 \
                    int idx = remaining % grid_len.a[i];                                                                                                                         \
                    remaining /= grid_len.a[i];                                                                                                                                  \
                    src_offset += idx * src_grid_stride.a[i];                                                                                                                    \
                    dst_offset += idx * dst_grid_stride.a[i];                                                                                                                    \
                    if (constrain_num > 0) {                                                                                                                                     \
                        for (int j = 0; j < constrain_num; j++) {                                                                                                                \
                            if (i == constrains.a[j].grid_idx) {                                                                                                                 \
                                constrains_grid_idx_multiple[j] = idx * constrains.a[j].grid_div_block;                                                                          \
                            }                                                                                                                                                    \
                        }                                                                                                                                                        \
                    }                                                                                                                                                            \
                }                                                                                                                                                                \
                                                                                                                                                                                 \
                /* 将结果存入共享内存 */                                                                                                                                \
                shared_src_offset = src_offset;                                                                                                                                  \
                shared_dst_offset = dst_offset;                                                                                                                                  \
                for (int j = 0; j < constrain_num; j++) {                                                                                                                        \
                    shared_constrains_grid_idx_multiple[j] = constrains_grid_idx_multiple[j];                                                                                    \
                }                                                                                                                                                                \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            /* 确保所有线程都能看到共享内存中的值 */                                                                                                            \
            __syncthreads();                                                                                                                                                     \
                                                                                                                                                                                 \
            /* 所有线程直接使用计算好的偏移值 */                                                                                                                  \
            int src_offset = shared_src_offset;                                                                                                                                  \
            int dst_offset = shared_dst_offset;                                                                                                                                  \
            int constrains_grid_idx_multiple[constrain_num > 0 ? constrain_num : 1];                                                                                             \
            for (int j = 0; j < constrain_num; j++) {                                                                                                                            \
                constrains_grid_idx_multiple[j] = shared_constrains_grid_idx_multiple[j];                                                                                        \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            for (int i = block_array_size - 1; i > 0; i--) {                                                                                                                     \
                int idx = remaining % block_len.a[i];                                                                                                                            \
                remaining /= block_len.a[i];                                                                                                                                     \
                /* 计算偏移量 */                                                                                                                                            \
                src_offset += idx * src_block_stride.a[i];                                                                                                                       \
                dst_offset += idx * dst_block_stride.a[i];                                                                                                                       \
                if (constrain_num > 0) {                                                                                                                                         \
                    for (int j = 0; j < constrain_num; j++) {                                                                                                                    \
                        if (i == constrains.a[j].block_idx) {                                                                                                                    \
                            if (constrains_grid_idx_multiple[j] + idx >= constrains.a[j].total_len) {                                                                            \
                                return;                                                                                                                                          \
                            }                                                                                                                                                    \
                        }                                                                                                                                                        \
                    }                                                                                                                                                            \
                }                                                                                                                                                                \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            src_offset += remaining * src_block_stride.a[0];                                                                                                                     \
            dst_offset += remaining * dst_block_stride.a[0];                                                                                                                     \
            for (int j = 0; j < constrain_num; j++) {                                                                                                                            \
                if (0 == constrains.a[j].block_idx) {                                                                                                                            \
                    if (constrains_grid_idx_multiple[j] + remaining >= constrains.a[j].total_len) {                                                                              \
                        return;                                                                                                                                                  \
                    }                                                                                                                                                            \
                }                                                                                                                                                                \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            /* 执行数据拷贝，注意offset已经是字节偏移 */                                                                                                         \
            *reinterpret_cast<Tmem_type *>(reinterpret_cast<char *>(dst) + dst_offset) = *reinterpret_cast<const Tmem_type *>(reinterpret_cast<const char *>(src) + src_offset); \
                                                                                                                                                                                 \
        } else {                                                                                                                                                                 \
            if (threadIdx.x == 0) { /* 只让0号线程计算 */                                                                                                                 \
                /* 计算当前block处理的数据在src和dst中的基础偏移(bytes) */                                                                                      \
                int src_offset = 0;                                                                                                                                              \
                int dst_offset = 0;                                                                                                                                              \
                int remaining = blockIdx.x;                                                                                                                                      \
                                                                                                                                                                                 \
                for (int i = grid_array_size - 1; i >= 0; i--) {                                                                                                                 \
                    int idx = remaining % grid_len.a[i];                                                                                                                         \
                    remaining /= grid_len.a[i];                                                                                                                                  \
                    src_offset += idx * src_grid_stride.a[i];                                                                                                                    \
                    dst_offset += idx * dst_grid_stride.a[i];                                                                                                                    \
                }                                                                                                                                                                \
                                                                                                                                                                                 \
                /* 将结果存入共享内存 */                                                                                                                                \
                shared_src_offset = src_offset;                                                                                                                                  \
                shared_dst_offset = dst_offset;                                                                                                                                  \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            /* 确保所有线程都能看到共享内存中的值 */                                                                                                            \
            __syncthreads();                                                                                                                                                     \
                                                                                                                                                                                 \
            /* 所有线程直接使用计算好的偏移值 */                                                                                                                  \
            int src_offset = shared_src_offset;                                                                                                                                  \
            int dst_offset = shared_dst_offset;                                                                                                                                  \
                                                                                                                                                                                 \
            for (int i = block_array_size - 1; i > 0; i--) {                                                                                                                     \
                int idx = remaining % block_len.a[i];                                                                                                                            \
                remaining /= block_len.a[i];                                                                                                                                     \
                /* 计算偏移量 */                                                                                                                                            \
                src_offset += idx * src_block_stride.a[i];                                                                                                                       \
                dst_offset += idx * dst_block_stride.a[i];                                                                                                                       \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            src_offset += remaining * src_block_stride.a[0];                                                                                                                     \
            dst_offset += remaining * dst_block_stride.a[0];                                                                                                                     \
                                                                                                                                                                                 \
            /* 执行数据拷贝，注意offset已经是字节偏移 */                                                                                                         \
            *reinterpret_cast<Tmem_type *>(reinterpret_cast<char *>(dst) + dst_offset) = *reinterpret_cast<const Tmem_type *>(reinterpret_cast<const char *>(src) + src_offset); \
        }                                                                                                                                                                        \
    }

// 定义支持的约束条件数量组合
#define DEFINE_KERNELS_BY_CONSTRAIN(block_array_size, grid_array_size) \
    DEFINE_KERNELS_BY_TYPE(0, block_array_size, grid_array_size)       \
    DEFINE_KERNELS_BY_TYPE(1, block_array_size, grid_array_size)       \
    DEFINE_KERNELS_BY_TYPE(2, block_array_size, grid_array_size)

// 定义支持的类型
#define DEFINE_KERNELS_BY_TYPE(constrain_num, block_array_size, grid_array_size)      \
    DEFINE_REARRANGE_KERNEL(uchar1, constrain_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(uchar2, constrain_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(float1, constrain_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(float2, constrain_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(float4, constrain_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(double4, constrain_num, block_array_size, grid_array_size)

#define MAX_BLOCK_ARRAY_SIZE 5

#define MAX_GRID_ARRAY_SIZE 5

// 为1-5和1-5的所有组合生成内核
DEFINE_KERNELS_BY_CONSTRAIN(1, 1)
DEFINE_KERNELS_BY_CONSTRAIN(1, 2)
DEFINE_KERNELS_BY_CONSTRAIN(1, 3)
DEFINE_KERNELS_BY_CONSTRAIN(1, 4)
DEFINE_KERNELS_BY_CONSTRAIN(1, 5)
DEFINE_KERNELS_BY_CONSTRAIN(2, 1)
DEFINE_KERNELS_BY_CONSTRAIN(2, 2)
DEFINE_KERNELS_BY_CONSTRAIN(2, 3)
DEFINE_KERNELS_BY_CONSTRAIN(2, 4)
DEFINE_KERNELS_BY_CONSTRAIN(2, 5)
DEFINE_KERNELS_BY_CONSTRAIN(3, 1)
DEFINE_KERNELS_BY_CONSTRAIN(3, 2)
DEFINE_KERNELS_BY_CONSTRAIN(3, 3)
DEFINE_KERNELS_BY_CONSTRAIN(3, 4)
DEFINE_KERNELS_BY_CONSTRAIN(3, 5)
DEFINE_KERNELS_BY_CONSTRAIN(4, 1)
DEFINE_KERNELS_BY_CONSTRAIN(4, 2)
DEFINE_KERNELS_BY_CONSTRAIN(4, 3)
DEFINE_KERNELS_BY_CONSTRAIN(4, 4)
DEFINE_KERNELS_BY_CONSTRAIN(4, 5)
DEFINE_KERNELS_BY_CONSTRAIN(5, 1)
DEFINE_KERNELS_BY_CONSTRAIN(5, 2)
DEFINE_KERNELS_BY_CONSTRAIN(5, 3)
DEFINE_KERNELS_BY_CONSTRAIN(5, 4)
DEFINE_KERNELS_BY_CONSTRAIN(5, 5)

// 准备参数结构体
struct RearrangeParams {
    std::vector<int> block_len;
    std::vector<int> src_block_stride;
    std::vector<int> dst_block_stride;
    std::vector<int> grid_len;
    std::vector<int> src_grid_stride;
    std::vector<int> dst_grid_stride;
    unsigned int block_dim;
    unsigned int block_len_total;
    std::vector<Constrains<int>> constrains;
    unsigned int unit_size;
};

void *get_rearrange_kernel(const RearrangeParams &params) {
    auto grid_num = params.grid_len.size();
    auto block_num = params.block_len.size();
    auto constrain_num = params.constrains.size();
    auto unit_size = params.unit_size;

    // 检查参数是否在支持的范围内
    if (grid_num > MAX_GRID_ARRAY_SIZE || grid_num == 0) {
        return nullptr;
    }

    if (block_num > MAX_BLOCK_ARRAY_SIZE || block_num == 0) {
        return nullptr;
    }

    if (constrain_num > 2) {
        return nullptr;
    }

    auto block_len = params.block_len.data();
    auto src_block_stride = params.src_block_stride.data();
    auto dst_block_stride = params.dst_block_stride.data();
    auto grid_len = params.grid_len.data();
    auto src_grid_stride = params.src_grid_stride.data();
    auto dst_grid_stride = params.dst_grid_stride.data();
    auto constrain = params.constrains.data();

    void *kernel_func = nullptr;
#define GET_REARRANGE_KERNEL(Tmem_type, block_array_size, grid_array_size, constrain_num) \
    kernel_func = (void *)rearrange_unit_##Tmem_type##_block_##block_array_size##_grid_##grid_array_size##_constrain_##constrain_num;

#define GET_REARRANGE_KERNEL_BY_TYPE(block_array_size, grid_array_size, constrain_num)   \
    switch (unit_size) {                                                                 \
    case 1:                                                                              \
        GET_REARRANGE_KERNEL(uchar1, block_array_size, grid_array_size, constrain_num);  \
        break;                                                                           \
    case 2:                                                                              \
        GET_REARRANGE_KERNEL(uchar2, block_array_size, grid_array_size, constrain_num);  \
        break;                                                                           \
    case 4:                                                                              \
        GET_REARRANGE_KERNEL(float1, block_array_size, grid_array_size, constrain_num);  \
        break;                                                                           \
    case 8:                                                                              \
        GET_REARRANGE_KERNEL(float2, block_array_size, grid_array_size, constrain_num);  \
        break;                                                                           \
    case 16:                                                                             \
        GET_REARRANGE_KERNEL(float4, block_array_size, grid_array_size, constrain_num);  \
        break;                                                                           \
    case 32:                                                                             \
        GET_REARRANGE_KERNEL(double4, block_array_size, grid_array_size, constrain_num); \
        break;                                                                           \
    default:                                                                             \
        return nullptr;                                                                  \
    }

#define GET_REARRANGE_KERNEL_BY_CONSTRAIN(block_array_size, grid_array_size) \
    switch (constrain_num) {                                                 \
    case 0:                                                                  \
        GET_REARRANGE_KERNEL_BY_TYPE(block_array_size, grid_array_size, 0);  \
        break;                                                               \
    case 1:                                                                  \
        GET_REARRANGE_KERNEL_BY_TYPE(block_array_size, grid_array_size, 1);  \
        break;                                                               \
    case 2:                                                                  \
        GET_REARRANGE_KERNEL_BY_TYPE(block_array_size, grid_array_size, 2);  \
        break;                                                               \
    }

#define GET_REARRANGE_KERNEL_BY_GRID_NUM(block_array_size)      \
    switch (grid_num) {                                         \
    case 1:                                                     \
        GET_REARRANGE_KERNEL_BY_CONSTRAIN(block_array_size, 1); \
        break;                                                  \
    case 2:                                                     \
        GET_REARRANGE_KERNEL_BY_CONSTRAIN(block_array_size, 2); \
        break;                                                  \
    case 3:                                                     \
        GET_REARRANGE_KERNEL_BY_CONSTRAIN(block_array_size, 3); \
        break;                                                  \
    case 4:                                                     \
        GET_REARRANGE_KERNEL_BY_CONSTRAIN(block_array_size, 4); \
        break;                                                  \
    case 5:                                                     \
        GET_REARRANGE_KERNEL_BY_CONSTRAIN(block_array_size, 5); \
        break;                                                  \
    }

#define GET_REARRANGE_KERNEL_BY_BLOCK_NUM    \
    switch (block_num) {                     \
    case 1:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(1); \
        break;                               \
    case 2:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(2); \
        break;                               \
    case 3:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(3); \
        break;                               \
    case 4:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(4); \
        break;                               \
    case 5:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(5); \
        break;                               \
    }

    GET_REARRANGE_KERNEL_BY_BLOCK_NUM

    return kernel_func;
}
#endif // __REARRANGE_CUDA_KERNEL_H__
