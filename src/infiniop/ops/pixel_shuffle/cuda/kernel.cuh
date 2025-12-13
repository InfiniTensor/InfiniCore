#ifndef __PIXEL_SHUFFLE_KERNEL_CUH__
#define __PIXEL_SHUFFLE_KERNEL_CUH__

#include <cstdio>

template <typename T>
__global__ void pixel_shuffle_kernel(
    const T* input,
    T* output,
    int r,
    int B, int C_out, int H_out, int W_out,
    // 使用 int/size_t 都可以，但确保它们是元素步长
    size_t in_b_stride, size_t in_c_stride,
    size_t in_h_stride, size_t in_w_stride,
    size_t out_b_stride, size_t out_c_stride,
    size_t out_h_stride, size_t out_w_stride)
{
    // 1. 计算输出坐标
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;

    // --- DEBUG TRAP START: 检查 Kernel 启动 ---
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // 只有第一个 Block 的第一个 Thread 打印
        printf("[KERNEL START] Type size: %zu bytes. Stride B: %zu\n", sizeof(T), in_b_stride);
    }
    // --- DEBUG TRAP END ---

    if (w >= W_out || h >= H_out) return;

    int c = bc % C_out;
    int b = bc / C_out;
    if (b >= B) return;

    // 2. 映射到输入坐标
    int dh = h % r;
    int dw = w % r;
    int h_in = h / r;
    int w_in = w / r;
    int c_in = c * r * r + dh * r + dw;

    // 3. 计算内存偏移 (使用 size_t)
    // size_t 保证了 64 位，能够处理更大的内存地址
    size_t in_offset =
        (size_t)b * in_b_stride +
        (size_t)c_in * in_c_stride +
        (size_t)h_in * in_h_stride +
        (size_t)w_in * in_w_stride;

    size_t out_offset =
        (size_t)b * out_b_stride +
        (size_t)c * out_c_stride +
        (size_t)h * out_h_stride +
        (size_t)w * out_w_stride;
    
    // 4. 数据搬运
    output[out_offset] = input[in_offset];
}

#endif // __PIXEL_SHUFFLE_KERNEL_CUH__