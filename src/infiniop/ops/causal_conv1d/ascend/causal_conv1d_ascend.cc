#include "causal_conv1d_ascend.h"
#include "../../../devices/ascend/ascend_handle.h"

namespace op::causal_conv1d::ascend {
struct Descriptor::Opaque {
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t conv_state_desc,
    infiniopTensorDescriptor_t final_conv_state_desc,
    infiniopTensorDescriptor_t qkv_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t cu_seqlens_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc) {
    auto result = CausalConv1dInfo::create(
        out_desc, conv_state_desc, final_conv_state_desc, qkv_desc, weight_desc,
        bias_desc, cu_seqlens_desc, initial_state_indices_desc,
        final_state_indices_desc);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(
        new Opaque{}, result.take(), 0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size, void *out, void *conv_state,
    void *final_conv_state, const void *qkv, const void *weight,
    const void *bias, const void *cu_seqlens,
    const void *initial_state_indices, const void *final_state_indices,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    (void)workspace;
    return causal_conv1d_kernel_launch(
        out, conv_state, final_conv_state, qkv, weight, bias, cu_seqlens,
        initial_state_indices, final_state_indices, _info.data_dtype,
        _info.has_bias, _info.has_cu_seqlens,
        _info.cu_seqlens_dtype == INFINI_DTYPE_I64,
        _info.initial_state_indices_dtype == INFINI_DTYPE_I64,
        _info.final_state_indices_dtype == INFINI_DTYPE_I64,
        _info.indexed_state_pool, _info.request_count, _info.T, _info.C,
        _info.total_tokens, _info.pool_size,
        _info.out_strides[0], _info.out_strides[1], _info.out_strides[2],
        _info.conv_state_strides[0], _info.conv_state_strides[1],
        _info.conv_state_strides[2],
        _info.final_conv_state_strides.empty() ? 0 : _info.final_conv_state_strides[0],
        _info.final_conv_state_strides.empty() ? 0 : _info.final_conv_state_strides[1],
        _info.final_conv_state_strides.empty() ? 0 : _info.final_conv_state_strides[2],
        _info.qkv_strides[0], _info.qkv_strides[1], _info.qkv_strides[2],
        _info.weight_strides[0], _info.weight_strides[2],
        _info.bias_strides.empty() ? 0 : _info.bias_strides[0], stream);
}
} // namespace op::causal_conv1d::ascend
