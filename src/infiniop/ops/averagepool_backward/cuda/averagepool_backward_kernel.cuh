#ifndef __AVERAGEPOOL_BACKWARD_KERNEL_H__
#define __AVERAGEPOOL_BACKWARD_KERNEL_H__

#include <cmath>

template <typename T>
__global__ void
avgpool1d_pytorch_backward_kernel(const T *grad_output, T *grad_input,
                                  int batch_size, int channels,
                                  int input_length, int output_length,
                                  int kernel_size, int stride, int padding) {

    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = blockIdx.z * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || channel_idx >= channels || output_idx >= output_length) {
        return;
    }

    const T *grad_output_ptr = grad_output + batch_idx * channels * output_length + channel_idx * output_length;
    T *grad_input_ptr = grad_input + batch_idx * channels * input_length + channel_idx * input_length;

    // 从输出中获取梯度值
    float grad = static_cast<float>(grad_output_ptr[output_idx]);
    int window_start = output_idx * stride - padding;

    int pool_size = 0;
    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = window_start + k;
        if ((input_pos >= 0 && input_pos < input_length) || (input_pos >= -padding && input_pos < input_length + padding)) {
            pool_size++;
        }
    }

    // 避免除以零的极端情况
    if (pool_size == 0) {
        return;
    }

    float grad_per_input = grad / static_cast<float>(pool_size);
    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = window_start + k;
        if (input_pos >= 0 && input_pos < input_length) {
            // Atomically add the distributed gradient to the input gradient tensor
            atomicAdd(&grad_input_ptr[input_pos], static_cast<T>(grad_per_input));
        }
    }
}

template <typename T>
__global__ void avgpool2d_pytorch_backward_kernel(
    const T *grad_output, T *grad_input, int batch_size, int channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w) {

    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = blockIdx.z * blockDim.x + threadIdx.x;

    int total_output_elements = output_height * output_width;
    if (batch_idx >= batch_size || channel_idx >= channels || output_idx >= total_output_elements) {
        return;
    }

    // 将线性输出索引转换为二维坐标
    int out_h = output_idx / output_width;
    int out_w = output_idx % output_width;

    const T *grad_output_ptr = grad_output + batch_idx * channels * total_output_elements + channel_idx * total_output_elements;
    T *grad_input_ptr = grad_input + batch_idx * channels * input_height * input_width + channel_idx * input_height * input_width;

    float grad = static_cast<float>(grad_output_ptr[output_idx]);
    int window_start_h = out_h * stride_h - pad_h;
    int window_start_w = out_w * stride_w - pad_w;

    int pool_size = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int input_h = window_start_h + kh;
            int input_w = window_start_w + kw;
            if ((input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) || (input_h >= -pad_h && input_h < input_height + pad_h && input_w >= -pad_w && input_w < input_width + pad_w)) {
                pool_size++;
            }
        }
    }

    if (pool_size == 0) {
        return;
    }

    float grad_per_input = grad / static_cast<float>(pool_size);

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int input_h = window_start_h + kh;
            int input_w = window_start_w + kw;

            if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                int input_idx = input_h * input_width + input_w;
                atomicAdd(&grad_input_ptr[input_idx], static_cast<T>(grad_per_input));
            }
        }
    }
}

template <typename T>
__global__ void avgpool3d_pytorch_backward_kernel(
    const T *grad_output, T *grad_input, int batch_size, int channels,
    int input_depth, int input_height, int input_width, int output_depth,
    int output_height, int output_width, int kernel_d, int kernel_h,
    int kernel_w, int stride_d, int stride_h, int stride_w, int pad_d,
    int pad_h, int pad_w) {

    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = blockIdx.z * blockDim.x + threadIdx.x;

    int total_output_elements = output_depth * output_height * output_width;
    if (batch_idx >= batch_size || channel_idx >= channels || output_idx >= total_output_elements) {
        return;
    }

    // 将线性输出索引转换为三维坐标
    int out_d = output_idx / (output_height * output_width);
    int remaining = output_idx % (output_height * output_width);
    int out_h = remaining / output_width;
    int out_w = remaining % output_width;

    int input_spatial_size = input_depth * input_height * input_width;
    const T *grad_output_ptr = grad_output + batch_idx * channels * total_output_elements + channel_idx * total_output_elements;
    T *grad_input_ptr = grad_input + batch_idx * channels * input_spatial_size + channel_idx * input_spatial_size;

    float grad = static_cast<float>(grad_output_ptr[output_idx]);
    int window_start_d = out_d * stride_d - pad_d;
    int window_start_h = out_h * stride_h - pad_h;
    int window_start_w = out_w * stride_w - pad_w;

    int pool_size = 0;
    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_d = window_start_d + kd;
                int input_h = window_start_h + kh;
                int input_w = window_start_w + kw;

                if ((input_d >= 0 && input_d < input_depth && input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) || (input_d >= -pad_d && input_d < input_depth + pad_d && input_h >= -pad_h && input_h < input_height + pad_h && input_w >= -pad_w && input_w < input_width + pad_w)) {
                    pool_size++;
                }
            }
        }
    }

    if (pool_size == 0) {
        return;
    }

    float grad_per_input = grad / static_cast<float>(pool_size);

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_d = window_start_d + kd;
                int input_h = window_start_h + kh;
                int input_w = window_start_w + kw;

                if (input_d >= 0 && input_d < input_depth && input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                    int input_idx = (input_d * input_height + input_h) * input_width + input_w;
                    atomicAdd(&grad_input_ptr[input_idx], static_cast<T>(grad_per_input));
                }
            }
        }
    }
}

#endif // __AVERAGEPOOL_BACKWARD_KERNEL_H__
