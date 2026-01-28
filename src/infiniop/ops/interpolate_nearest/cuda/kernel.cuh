#ifndef INTERPOLATE_NEAREST_KERNEL_CUH
#define INTERPOLATE_NEAREST_KERNEL_CUH

#include "../info.h"
#include <cmath>

template <typename T>
__device__ inline size_t
compute_input_index_1d(size_t idx, const InterpolateNearestInfo &info) {
    size_t temp = idx;

    // 1D 插值：3D 张量 (N, C, W)
    size_t w = temp % info.output_size[0]; // width 在索引 0
    temp /= info.output_size[0];
    size_t c = temp % info.channels;
    size_t b = temp / info.channels;

    float inv_scale = static_cast<float>(info.input_size[0]) / static_cast<float>(info.output_size[0]);
    size_t input_w = min(static_cast<size_t>(floorf(static_cast<float>(w) * inv_scale)),
                         info.input_size[0] - 1);

    return b * info.input_stride[0] + c * info.input_stride[1] + input_w * info.input_stride[2];
}

template <typename T>
__device__ inline size_t
compute_input_index_2d(size_t idx, const InterpolateNearestInfo &info) {
    size_t temp = idx;

    // 2D 插值：4D 张量 (N, C, H, W)
    size_t w = temp % info.output_size[1]; // width 在索引 1
    temp /= info.output_size[1];
    size_t h = temp % info.output_size[0]; // height 在索引 0
    temp /= info.output_size[0];
    size_t c = temp % info.channels;
    size_t b = temp / info.channels;

    float inv_scale_h = static_cast<float>(info.input_size[0]) / static_cast<float>(info.output_size[0]);
    float inv_scale_w = static_cast<float>(info.input_size[1]) / static_cast<float>(info.output_size[1]);

    size_t input_h = min(static_cast<size_t>(floorf(static_cast<float>(h) * inv_scale_h)),
                         info.input_size[0] - 1);
    size_t input_w = min(static_cast<size_t>(floorf(static_cast<float>(w) * inv_scale_w)),
                         info.input_size[1] - 1);

    return b * info.input_stride[0] + c * info.input_stride[1] + input_h * info.input_stride[2] + input_w * info.input_stride[3];
}

template <typename T>
__device__ inline size_t
compute_input_index_3d(size_t idx, const InterpolateNearestInfo &info) {
    size_t temp = idx;

    // 3D 插值：5D 张量 (N, C, D, H, W)
    size_t w = temp % info.output_size[2]; // width 在索引 2
    temp /= info.output_size[2];
    size_t h = temp % info.output_size[1]; // height 在索引 1
    temp /= info.output_size[1];
    size_t d = temp % info.output_size[0]; // depth 在索引 0
    temp /= info.output_size[0];
    size_t c = temp % info.channels;
    size_t b = temp / info.channels;

    float inv_scale_d = static_cast<float>(info.input_size[0]) / static_cast<float>(info.output_size[0]);
    float inv_scale_h = static_cast<float>(info.input_size[1]) / static_cast<float>(info.output_size[1]);
    float inv_scale_w = static_cast<float>(info.input_size[2]) / static_cast<float>(info.output_size[2]);

    size_t input_d = min(static_cast<size_t>(floorf(static_cast<float>(d) * inv_scale_d)),
                         info.input_size[0] - 1);
    size_t input_h = min(static_cast<size_t>(floorf(static_cast<float>(h) * inv_scale_h)),
                         info.input_size[1] - 1);
    size_t input_w = min(static_cast<size_t>(floorf(static_cast<float>(w) * inv_scale_w)),
                         info.input_size[2] - 1);

    return b * info.input_stride[0] + c * info.input_stride[1] + input_d * info.input_stride[2] + input_h * info.input_stride[3] + input_w * info.input_stride[4];
}

template <typename T>
__device__ inline size_t
compute_output_index(size_t idx, const InterpolateNearestInfo &info) {
    size_t temp = idx;
    size_t w, h, d, c, b;

    switch (info.dim) {
    case INTERPOLATE_1D: {
        // 3D 张量 (N, C, W)
        w = temp % info.output_size[0];
        temp /= info.output_size[0];
        c = temp % info.channels;
        b = temp / info.channels;
        return b * info.output_stride[0] + c * info.output_stride[1] + w * info.output_stride[2];
    }

    case INTERPOLATE_2D: {
        // 4D 张量 (N, C, H, W)
        w = temp % info.output_size[1];
        temp /= info.output_size[1];
        h = temp % info.output_size[0];
        temp /= info.output_size[0];
        c = temp % info.channels;
        b = temp / info.channels;
        return b * info.output_stride[0] + c * info.output_stride[1] + h * info.output_stride[2] + w * info.output_stride[3];
    }

    case INTERPOLATE_3D: {
        // 5D 张量 (N, C, D, H, W)
        w = temp % info.output_size[2];
        temp /= info.output_size[2];
        h = temp % info.output_size[1];
        temp /= info.output_size[1];
        d = temp % info.output_size[0];
        temp /= info.output_size[0];
        c = temp % info.channels;
        b = temp / info.channels;
        return b * info.output_stride[0] + c * info.output_stride[1] + d * info.output_stride[2] + h * info.output_stride[3] + w * info.output_stride[4];
    }

    default:
        return 0;
    }
}

__host__ __device__ inline size_t
calculate_total_elements(const InterpolateNearestInfo &info) {
    size_t total = info.batch_size * info.channels;
    switch (info.dim) {
    case INTERPOLATE_1D:
        total *= info.output_size[0]; // width
        break;
    case INTERPOLATE_2D:
        total *= info.output_size[0] * info.output_size[1]; // height * width
        break;
    case INTERPOLATE_3D:
        total *= info.output_size[0] * info.output_size[1] * info.output_size[2]; // depth * height * width
        break;
    }
    return total;
}

template <typename T>
__global__ void interpolate_nearest_kernel(T *output, const T *input,
                                           InterpolateNearestInfo info) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = calculate_total_elements(info);

    if (idx < total_elements) {
        size_t input_idx;

        switch (info.dim) {
        case INTERPOLATE_1D:
            input_idx = compute_input_index_1d<T>(idx, info);
            break;
        case INTERPOLATE_2D:
            input_idx = compute_input_index_2d<T>(idx, info);
            break;
        case INTERPOLATE_3D:
            input_idx = compute_input_index_3d<T>(idx, info);
            break;
        default:
            return;
        }

        size_t output_idx = compute_output_index<T>(idx, info);
        output[output_idx] = input[input_idx];
    }
}

#endif // INTERPOLATE_NEAREST_KERNEL_CUH
