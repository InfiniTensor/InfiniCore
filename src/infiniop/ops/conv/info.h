#ifndef __CONV_INFO_H__
#define __CONV_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "../../devices/cuda/cuda_handle.cuh"
#include <algorithm>
namespace op::conv {

class CudnnConvHandler{
    CudnnConvHandler() = default;
public:
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;

    ~CudnnConvHandler() {
        if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
        if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
        if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
        if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
    }
    static infiniStatus_t create(
        ConvInfo& info,
        infiniDtype_t data_type,
        cudnnDataType_t compute_type) {

        CudnnConvHandler& handler = *info.handler;

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&handler.x_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&handler.y_desc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&handler.w_desc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&handler.conv_desc));
        
        cudnnDataType_t x_type, y_type, w_type;
        if (data_type == INFINI_DTYPE_F16) {
            x_type = y_type = w_type = CUDNN_DATA_HALF;
        } else if (data_type == INFINI_DTYPE_F32) {
            x_type = y_type = w_type = CUDNN_DATA_FLOAT;
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        size_t n_dim = info.ndim + 2;
        
        // 准备维度和步长数组
        std::vector<int> input_dims(n_dim);
        std::vector<int> output_dims(n_dim);
        std::vector<int> filter_dims(n_dim);
        std::vector<int> input_strides(n_dim);
        std::vector<int> output_strides(n_dim);
        
        // 设置批次和通道维度
        input_dims[0] = info.batch;
        input_dims[1] = info.in_channels;
        output_dims[0] = info.batch;
        output_dims[1] = info.out_channels;
        filter_dims[0] = info.out_channels;
        filter_dims[1] = info.in_channels;
        
        // 设置空间维度
        for (size_t i = 0; i < info.ndim; ++i) {
            input_dims[i + 2] = info.input_dims[i];
            output_dims[i + 2] = info.output_dims[i];
            filter_dims[i + 2] = info.kernel_dims[i];
        }
        
        // 计算步长
        input_strides[n_dim - 1] = 1;
        output_strides[n_dim - 1] = 1;
        for (int i = n_dim - 2; i >= 0; --i) {
            input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
            output_strides[i] = output_strides[i + 1] * output_dims[i + 1];
        }
        
        // 设置描述符
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(
            handler.x_desc, x_type, n_dim, input_dims.data(), input_strides.data()));
            
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(
            handler.y_desc, y_type, n_dim, output_dims.data(), output_strides.data()));
            
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(
            handler.w_desc, w_type, CUDNN_TENSOR_NCHW, n_dim, filter_dims.data()));
        
        // 准备卷积参数
        std::vector<int> pads(info.ndim);
        std::vector<int> strides(info.ndim);
        std::vector<int> dilations(info.ndim);
        
        for (size_t i = 0; i < info.ndim; ++i) {
            pads[i] = info.pads_info[i];
            strides[i] = info.strides_info[i];
            dilations[i] = info.dilations_info[i];
        }
        
        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
            handler.conv_desc, info.ndim, pads.data(), strides.data(), dilations.data(),
            CUDNN_CROSS_CORRELATION, compute_type));
    }

};

class ConvInfo {
    ConvInfo() = default;
public:
    size_t ndim;
    size_t batch;
    size_t in_channels;
    size_t out_channels;
    std::vector<size_t> input_dims;
    std::vector<size_t> kernel_dims;
    std::vector<size_t> output_dims;  
    std::vector<size_t> pads_info;      
    std::vector<size_t> strides_info;   
    std::vector<size_t> dilations_info; 
    std::shared_ptr<CudnnConvHandler> handler = nullptr;
    static utils::Result<ConvInfo> create(
        infiniopHandle_t handle_,
        infiniopTensorDescriptor_t y_desc, 
        infiniopTensorDescriptor_t x_desc, 
        infiniopTensorDescriptor_t w_desc,
        const void *pads,
        const void *strides,
        const void *dilations,
        size_t n) {
        auto dtype = y_desc->dtype();
        if (dtype != x_desc->dtype() || dtype != w_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);
        
        ConvInfo info;

        info.ndim = n;
        size_t new_dims = n + 2;

        if (x_desc->ndim() < new_dims || y_desc->ndim() < new_dims || w_desc->ndim() < new_dims) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        info.batch = x_desc->shape()[0];
        info.in_channels = x_desc->shape()[1];
        info.out_channels = w_desc->shape()[0];

        if (y_desc->shape()[0] != info.batch || y_desc->shape()[1] != info.out_channels || w_desc->shape()[1] != info.in_channels) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        info.input_dims.resize(info.ndim);
        info.kernel_dims.resize(info.ndim);
        info.output_dims.resize(info.ndim);
        info.pads_info.resize(info.ndim);
        info.strides_info.resize(info.ndim);
        info.dilations_info.resize(info.ndim);

        const size_t *pads_info = reinterpret_cast<const size_t *>(pads);
        const size_t *strides_info = reinterpret_cast<const size_t *>(strides);
        const size_t *dilations_info = reinterpret_cast<const size_t *>(dilations);

        for (size_t i = 0; i < info.ndim; i++) {
            info.input_dims[i] = x_desc->shape()[i + 2];
            info.kernel_dims[i] = w_desc->shape()[i + 2];
            info.output_dims[i] = y_desc->shape()[i + 2];
            info.pads_info[i] = pads_info == nullptr ? 0 : pads_info[i];
            info.strides_info[i] = strides_info == nullptr ? 1 : strides_info[i];
            info.dilations_info[i] = dilations_info == nullptr ? 1 : dilations_info[i];
            size_t expected_output = (info.input_dims[i] + info.pads_info[i] * 2 - info.dilations_info[i] * (info.kernel_dims[i] - 1) - 1) / info.strides_info[i] + 1;
            if (info.output_dims[i] != expected_output) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
        if (handle_->device_id == 1) {
            info.handler = std::make_shared<CudnnConvHandler>();
            CHECK_CUDNN(CudnnConvHandler::create(info, dtype, CUDNN_DATA_FLOAT));
        } 
        return utils::Result<ConvInfo>(info);
    }
};

}

#endif // __CONV_INFO_H__

