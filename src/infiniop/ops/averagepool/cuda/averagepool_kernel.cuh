#ifndef __AVERAGEPOOL_KERNEL_H__
#define __AVERAGEPOOL_KERNEL_H__

#include <cmath>

// 1D平均池化kernel，兼容PyTorch的隐式填充逻辑
template <typename T>
__global__ void avgpool1d_pytorch_compatible_kernel(
    const T *input, T *output, int batch_size, int channels, int input_length,
    int output_length, int kernel_size, int stride, int padding) {

    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = blockIdx.z * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || channel_idx >= channels || output_idx >= output_length) {
        return;
    }

    // 计算输入和输出的偏移
    const T *input_ptr = input + batch_idx * channels * input_length + channel_idx * input_length;
    T *output_ptr = output + batch_idx * channels * output_length + channel_idx * output_length;

    // 计算池化窗口的起始位置
    int window_start = output_idx * stride - padding;

    // 使用单精度进行中间计算
    float sum = 0.0f;
    int valid_count = 0;

    // 遍历池化窗口
    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = window_start + k;

        if (input_pos >= 0 && input_pos < input_length) {
            // 有效的输入位置，转换为单精度进行累加
            sum += static_cast<float>(input_ptr[input_pos]);
            valid_count++;
        } else if (input_pos >= -padding && input_pos < input_length + padding) {
            // 显式填充区域，值为0，只增加计数
            valid_count++;
        }
        // 其他位置是隐式填充，不计入分母
    }

    // 计算平均值并转换回原始数据类型
    if (valid_count > 0) {
        float result = sum / static_cast<float>(valid_count);
        output_ptr[output_idx] = static_cast<T>(result);
    } else {
        output_ptr[output_idx] = T(0);
    }
}

// 2D平均池化kernel，兼容PyTorch的隐式填充逻辑
template <typename T>
__global__ void avgpool2d_pytorch_compatible_kernel(
    const T *input, T *output, int batch_size, int channels, int input_height,
    int input_width, int output_height, int output_width, int kernel_h,
    int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {

    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = blockIdx.z * blockDim.x + threadIdx.x;

    int total_output_elements = output_height * output_width;
    if (batch_idx >= batch_size || channel_idx >= channels || output_idx >= total_output_elements) {
        return;
    }

    // 将线性索引转换为2D坐标
    int out_h = output_idx / output_width;
    int out_w = output_idx % output_width;

    // 计算输入和输出的偏移
    const T *input_ptr = input + batch_idx * channels * input_height * input_width + channel_idx * input_height * input_width;
    T *output_ptr = output + batch_idx * channels * output_height * output_width + channel_idx * output_height * output_width;

    // 计算池化窗口的起始位置
    int window_start_h = out_h * stride_h - pad_h;
    int window_start_w = out_w * stride_w - pad_w;

    // 使用单精度进行中间计算
    float sum = 0.0f;
    int valid_count = 0;

    // 遍历池化窗口
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int input_h = window_start_h + kh;
            int input_w = window_start_w + kw;

            if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                // 有效的输入位置，转换为单精度进行累加
                int input_idx = input_h * input_width + input_w;
                sum += static_cast<float>(input_ptr[input_idx]);
                valid_count++;
            } else if (input_h >= -pad_h && input_h < input_height + pad_h && input_w >= -pad_w && input_w < input_width + pad_w) {
                // 显式填充区域，值为0，只增加计数
                valid_count++;
            }
            // 其他位置是隐式填充，不计入分母
        }
    }

    // 计算平均值并转换回原始数据类型
    if (valid_count > 0) {
        float result = sum / static_cast<float>(valid_count);
        output_ptr[output_idx] = static_cast<T>(result);
    } else {
        output_ptr[output_idx] = T(0);
    }
}

// 3D平均池化kernel，兼容PyTorch的隐式填充逻辑
template <typename T>
__global__ void avgpool3d_pytorch_compatible_kernel(
    const T *input, T *output, int batch_size, int channels, int input_depth,
    int input_height, int input_width, int output_depth, int output_height,
    int output_width, int kernel_d, int kernel_h, int kernel_w, int stride_d,
    int stride_h, int stride_w, int pad_d, int pad_h, int pad_w) {

    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = blockIdx.z * blockDim.x + threadIdx.x;

    int total_output_elements = output_depth * output_height * output_width;
    if (batch_idx >= batch_size || channel_idx >= channels || output_idx >= total_output_elements) {
        return;
    }

    // 将线性索引转换为3D坐标
    int out_d = output_idx / (output_height * output_width);
    int remaining = output_idx % (output_height * output_width);
    int out_h = remaining / output_width;
    int out_w = remaining % output_width;

    // 计算输入和输出的偏移
    int input_spatial_size = input_depth * input_height * input_width;
    int output_spatial_size = output_depth * output_height * output_width;

    const T *input_ptr = input + batch_idx * channels * input_spatial_size + channel_idx * input_spatial_size;
    T *output_ptr = output + batch_idx * channels * output_spatial_size + channel_idx * output_spatial_size;

    // 计算池化窗口的起始位置
    int window_start_d = out_d * stride_d - pad_d;
    int window_start_h = out_h * stride_h - pad_h;
    int window_start_w = out_w * stride_w - pad_w;

    // 使用单精度进行中间计算
    float sum = 0.0f;
    int valid_count = 0;

    // 遍历池化窗口
    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_d = window_start_d + kd;
                int input_h = window_start_h + kh;
                int input_w = window_start_w + kw;

                if (input_d >= 0 && input_d < input_depth && input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                    // 有效的输入位置，转换为单精度进行累加
                    int input_idx = (input_d * input_height + input_h) * input_width + input_w;
                    sum += static_cast<float>(input_ptr[input_idx]);
                    valid_count++;
                } else if (input_d >= -pad_d && input_d < input_depth + pad_d && input_h >= -pad_h && input_h < input_height + pad_h && input_w >= -pad_w && input_w < input_width + pad_w) {
                    // 显式填充区域，值为0，只增加计数
                    valid_count++;
                }
                // 其他位置是隐式填充，不计入分母
            }
        }
    }

    // 计算平均值并转换回原始数据类型
    if (valid_count > 0) {
        float result = sum / static_cast<float>(valid_count);
        output_ptr[output_idx] = static_cast<T>(result);
    } else {
        output_ptr[output_idx] = T(0);
    }
}

#endif // __AVERAGEPOOL_KERNEL_H__
