#ifndef __RWKV5_WKV_INFO_H__
#define __RWKV5_WKV_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::rwkv5_wkv {

class Rwkv5WkvInfo {
    Rwkv5WkvInfo() = default;

public:
    infiniDtype_t dtype;
    size_t batch;
    size_t seq_len;
    size_t hidden_size;
    size_t num_heads;
    size_t head_size;

    static utils::Result<Rwkv5WkvInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t receptance_desc,
        infiniopTensorDescriptor_t key_desc,
        infiniopTensorDescriptor_t value_desc,
        infiniopTensorDescriptor_t time_decay_desc,
        infiniopTensorDescriptor_t time_faaaa_desc,
        infiniopTensorDescriptor_t state_desc) {

        auto dtype = receptance_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (out_desc->dtype() != dtype || key_desc->dtype() != dtype || value_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (time_decay_desc->dtype() != dtype || time_faaaa_desc->dtype() != dtype || state_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (out_desc->ndim() != 3 || receptance_desc->ndim() != 3 || key_desc->ndim() != 3 || value_desc->ndim() != 3
            || time_decay_desc->ndim() != 2 || time_faaaa_desc->ndim() != 2 || state_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const auto &shape = receptance_desc->shape();
        if (out_desc->shape() != shape || key_desc->shape() != shape || value_desc->shape() != shape) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const auto &time_shape = time_decay_desc->shape();
        if (time_faaaa_desc->shape() != time_shape || time_shape.size() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t batch = shape[0];
        const size_t seq_len = shape[1];
        const size_t hidden_size = shape[2];
        const size_t num_heads = time_shape[0];
        const size_t head_size = time_shape[1];
        if (hidden_size != num_heads * head_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const auto &state_shape = state_desc->shape();
        if (state_shape[0] < batch || state_shape[1] != num_heads || state_shape[2] != head_size || state_shape[3] != head_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (out_desc->strides()[2] != 1 || receptance_desc->strides()[2] != 1 || key_desc->strides()[2] != 1
            || value_desc->strides()[2] != 1 || time_decay_desc->strides()[1] != 1 || time_faaaa_desc->strides()[1] != 1
            || state_desc->strides()[3] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<Rwkv5WkvInfo>(Rwkv5WkvInfo{
            dtype,
            batch,
            seq_len,
            hidden_size,
            num_heads,
            head_size});
    }
};

} // namespace op::rwkv5_wkv

#endif
