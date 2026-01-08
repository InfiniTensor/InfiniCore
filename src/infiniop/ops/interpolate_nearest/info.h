#ifndef __INTERPOLATE_NEAREST_INFO_H__
#define __INTERPOLATE_NEAREST_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <cstddef>

enum InterpolateDim {
    INTERPOLATE_1D = 1, // 3D 张量 (N, C, W)
    INTERPOLATE_2D = 2, // 4D 张量 (N, C, H, W)
    INTERPOLATE_3D = 3  // 5D 张量 (N, C, D, H, W)
};

struct InterpolateNearestInfo {
    size_t batch_size;
    size_t channels;

    // 输入和输出的空间维度大小
    size_t input_size[3];  // [depth/height/width] 根据维度使用不同数量
    size_t output_size[3]; // [depth/height/width] 根据维度使用不同数量

    InterpolateDim dim; // 插值维度：1D, 2D, 3D
    infiniDtype_t dtype;

    // 张量步长（最多支持 5D 张量）
    size_t input_stride[5];
    size_t output_stride[5];

    static infiniStatus_t create(
        InterpolateNearestInfo *info,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc) {

        // 检查数据类型
        if (input_desc->dtype() != output_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto input_shape = input_desc->shape();
        auto output_shape = output_desc->shape();
        auto input_stride = input_desc->strides();
        auto output_stride = output_desc->strides();

        // 根据张量维度确定插值类型
        if (input_desc->ndim() == 3 && output_desc->ndim() == 3) {
            // 1D 插值：3D 张量 (N, C, W)
            info->dim = INTERPOLATE_1D;
            info->batch_size = input_shape[0];
            info->channels = input_shape[1];
            info->input_size[0] = input_shape[2];   // width
            info->output_size[0] = output_shape[2]; // width

            // 检查 N,C 维度匹配
            if (input_shape[0] != output_shape[0] || input_shape[1] != output_shape[1]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            // 复制步长
            for (int i = 0; i < 3; ++i) {
                info->input_stride[i] = input_stride[i];
                info->output_stride[i] = output_stride[i];
            }

        } else if (input_desc->ndim() == 4 && output_desc->ndim() == 4) {
            // 2D 插值：4D 张量 (N, C, H, W)
            info->dim = INTERPOLATE_2D;
            info->batch_size = input_shape[0];
            info->channels = input_shape[1];
            info->input_size[0] = input_shape[2];   // height
            info->input_size[1] = input_shape[3];   // width
            info->output_size[0] = output_shape[2]; // height
            info->output_size[1] = output_shape[3]; // width

            // 检查 N,C 维度匹配
            if (input_shape[0] != output_shape[0] || input_shape[1] != output_shape[1]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            // 复制步长
            for (int i = 0; i < 4; ++i) {
                info->input_stride[i] = input_stride[i];
                info->output_stride[i] = output_stride[i];
            }

        } else if (input_desc->ndim() == 5 && output_desc->ndim() == 5) {
            // 3D 插值：5D 张量 (N, C, D, H, W)
            info->dim = INTERPOLATE_3D;
            info->batch_size = input_shape[0];
            info->channels = input_shape[1];
            info->input_size[0] = input_shape[2];   // depth
            info->input_size[1] = input_shape[3];   // height
            info->input_size[2] = input_shape[4];   // width
            info->output_size[0] = output_shape[2]; // depth
            info->output_size[1] = output_shape[3]; // height
            info->output_size[2] = output_shape[4]; // width

            // 检查 N,C 维度匹配
            if (input_shape[0] != output_shape[0] || input_shape[1] != output_shape[1]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            // 复制步长
            for (int i = 0; i < 5; ++i) {
                info->input_stride[i] = input_stride[i];
                info->output_stride[i] = output_stride[i];
            }

        } else {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        info->dtype = input_desc->dtype();
        return INFINI_STATUS_SUCCESS;
    }
};

#endif // __INTERPOLATE_NEAREST_INFO_H__
