#ifndef __WHERE_INDICES_KERNEL_CUH__
#define __WHERE_INDICES_KERNEL_CUH__

#include <cstddef>
#include <cuda_runtime.h>

namespace op::where::cuda {

// 阶段1: 标记 True 元素 (将 bool 转换为 int64_t: 1 或 0)
// 支持 strided tensor：使用线性索引转换为多维索引，然后使用 stride 计算内存偏移
template <typename Tidx>
__global__ void markTrueElements(
    Tidx *flags,              // 输出：每个元素是否为 True (1/0)
    const bool *cond,         // 输入：条件张量
    const size_t *shape,      // 输入：张量形状
    const ptrdiff_t *strides, // 输入：张量 stride
    size_t numel,
    int ndim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // 将线性索引转换为多维索引
        size_t remaining = idx;
        size_t offset = 0;
        for (int dim = ndim - 1; dim >= 0; --dim) {
            size_t dim_idx = remaining % shape[dim];
            offset += dim_idx * static_cast<size_t>(strides[dim]);
            remaining /= shape[dim];
        }
        flags[idx] = cond[offset] ? static_cast<Tidx>(1) : static_cast<Tidx>(0);
    }
}

// 阶段2: 收集每个维度的索引
// 对于 N 维张量，需要为每个维度收集索引
template <typename Tidx>
__global__ void collectIndices(
    Tidx **outputs,           // 输出：NDIM 个索引张量的指针数组（在设备上）
    const Tidx *flags,        // 输入：标记数组（前缀和后）
    const bool *cond,         // 输入：条件张量（未使用，可为 nullptr）
    const size_t *shape,      // 输入：张量形状（在设备上）
    const ptrdiff_t *strides, // 输入：张量 stride（在设备上，未在此 kernel 中使用）
    size_t numel,
    int ndim) {
    (void)cond;
    (void)strides;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // 通过 flags 判断该线性索引是否对应 True 元素：
        // 对于 True 元素，flags[idx] 会比前一个位置的值大 1
        Tidx curr = flags[idx];
        Tidx prev = (idx == 0) ? static_cast<Tidx>(0) : flags[idx - 1];
        if (curr == prev) {
            return;
        }

        // 计算当前元素在输出中的位置（使用前缀和结果）
        // flags[idx] 是 inclusive sum，所以 flags[idx] - 1 是当前元素在输出中的位置
        // 对于第一个元素（idx=0），如果它是 True，flags[0] = 1，所以 output_idx = 0
        Tidx output_idx = curr - 1;

        // 线性索引 -> 多维索引
        size_t remaining = idx;
        for (int dim = ndim - 1; dim >= 0; --dim) {
            size_t dim_idx = remaining % shape[dim];
            outputs[dim][output_idx] = static_cast<Tidx>(dim_idx);
            remaining /= shape[dim];
        }
    }
}

} // namespace op::where::cuda

#endif // __WHERE_INDICES_KERNEL_CUH__
