#include "mrope_ascend.h"
#include "../../../devices/ascend/ascend_handle.h"

namespace op::mrope::ascend {

extern "C" infiniStatus_t mrope_ascend_kernel_launch(
    void *q_out, void *k_out, const void *q, const void *k,
    const void *cos, const void *sin, const void *positions,
    infiniDtype_t data_type, bool positions_i64,
    size_t num_tokens, size_t num_q_heads, size_t num_kv_heads,
    size_t head_size, size_t rotary_dim, size_t half_rotary_dim,
    ptrdiff_t q_out_stride_token, ptrdiff_t q_out_stride_head,
    ptrdiff_t k_out_stride_token, ptrdiff_t k_out_stride_head,
    ptrdiff_t q_stride_token, ptrdiff_t q_stride_head,
    ptrdiff_t k_stride_token, ptrdiff_t k_stride_head,
    ptrdiff_t cos_stride_position, ptrdiff_t sin_stride_position,
    ptrdiff_t positions_stride_axis, ptrdiff_t positions_stride_token,
    size_t max_position_embeddings, size_t section_t,
    size_t section_h, size_t section_w,
    bool positions_has_axes, bool interleaved, void *stream);

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t q_out_desc,
    infiniopTensorDescriptor_t k_out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t cos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t positions_desc,
    int head_size,
    int rotary_dim,
    int section_t,
    int section_h,
    int section_w,
    bool interleaved) {

    auto result = MRoPEInfo::create(
        q_out_desc, k_out_desc, q_desc, k_desc, cos_desc, sin_desc,
        positions_desc, head_size, rotary_dim, section_t, section_h,
        section_w, interleaved);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(
        result.take(), 0, new Opaque{}, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *q_out,
    void *k_out,
    const void *q,
    const void *k,
    const void *cos,
    const void *sin,
    const void *positions,
    void *stream) const {

    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.data_type == INFINI_DTYPE_F64) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return mrope_ascend_kernel_launch(
        q_out, k_out, q, k, cos, sin, positions, _info.data_type,
        _info.position_type == INFINI_DTYPE_I64,
        _info.num_tokens, _info.num_q_heads, _info.num_kv_heads,
        _info.head_size, _info.rotary_dim, _info.half_rotary_dim,
        _info.q_out_stride_token, _info.q_out_stride_head,
        _info.k_out_stride_token, _info.k_out_stride_head,
        _info.q_stride_token, _info.q_stride_head,
        _info.k_stride_token, _info.k_stride_head,
        _info.cos_stride_position, _info.sin_stride_position,
        _info.positions_stride_axis, _info.positions_stride_token,
        _info.max_position_embeddings, _info.section_t,
        _info.section_h, _info.section_w, _info.positions_has_axes,
        _info.interleaved, stream);
}

} // namespace op::mrope::ascend
