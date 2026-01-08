#include "rope_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

namespace op::rope::ascend {

Descriptor::~Descriptor()
    = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc,
    infiniopRoPEAlgo_t algo) {
    auto handle_ascned = reinterpret_cast<device::ascend::Handle *>(handle);
    auto result = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc, algo);
    CHECK_RESULT(result);
    size_t workspace_size = 0;
    *desc_ptr = new Descriptor(std::move(result.take()), workspace_size, nullptr, handle_ascned->device, handle_ascned->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const {
    CHECK_DTYPE(_info.data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F16);

    auto data_type = _info.data_type;
    auto pos_type = _info.pos_type;
    auto seq_len = _info.seqlen;
    auto nhead = _info.nhead;
    auto dhead = _info.dhead;

    auto y_stride_seqlen = _info.y_stride_seqlen;
    auto y_stride_nhead = _info.y_stride_nhead;
    auto y_stride_batch = _info.y_stride_batch;
    auto x_stride_seqlen = _info.x_stride_seqlen;
    auto x_stride_nhead = _info.x_stride_nhead;
    auto x_stride_batch = _info.x_stride_batch;

    for (size_t b = 0; b < _info.batch; ++b) {
        auto y_offset = b * y_stride_batch;
        auto x_offset = b * x_stride_batch;
        infiniStatus_t status = rope_kernel_launch(
            static_cast<void *>(static_cast<char *>(y) + y_offset * infiniSizeOf(data_type)),
            (void *)(static_cast<const char *>(x) + x_offset * infiniSizeOf(data_type)),
            (void *)(static_cast<const char *>(pos_ids) + (_info.pos_has_batch_dim ? b * seq_len * infiniSizeOf(pos_type) : 0)),
            const_cast<void *>(sin_table),
            const_cast<void *>(cos_table),
            seq_len,
            nhead,
            dhead,
            data_type,
            pos_type,
            y_stride_seqlen,
            y_stride_nhead,
            x_stride_seqlen,
            x_stride_nhead,
            _info.algo,
            stream);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::rope::ascend
