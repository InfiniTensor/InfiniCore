#include "rope_kunlun.h"
#include "../../../devices/kunlun/kunlun_handle.h"
#include "../../../devices/kunlun/kunlun_type.h"
#include <memory>

void RoPEF32I32F32(void *destination, const void *source,
                   const void *pos_ids, const void *sin_table, const void *cos_table,
                   kunlun_size_t seqlen, kunlun_size_t nhead, kunlun_size_t dhead,
                   kunlun_ptrdiff_t x_stride_seqlen, kunlun_ptrdiff_t x_stride_nhead,
                   kunlun_ptrdiff_t y_stride_seqlen, kunlun_ptrdiff_t y_stride_nhead,
                   XPUStream stream);

namespace op::rope::kunlun {

struct Descriptor::Opaque {
    std::shared_ptr<device::kunlun::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc) {

    auto result = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc);
    CHECK_RESULT(result);

    // Create descriptor
    *desc_ptr = new Descriptor(
        result.take(),
        0,
        new Descriptor::Opaque{static_cast<device::kunlun::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

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
    kunlun_size_t seqlen = (kunlun_size_t)_info.seqlen;
    kunlun_size_t nhead = (kunlun_size_t)_info.nhead;
    kunlun_size_t dhead = (kunlun_size_t)_info.dhead;
    kunlun_ptrdiff_t x_stride_seqlen = (kunlun_ptrdiff_t)_info.x_stride_seqlen;
    kunlun_ptrdiff_t x_stride_nhead = (kunlun_ptrdiff_t)_info.x_stride_nhead;
    kunlun_ptrdiff_t y_stride_seqlen = (kunlun_ptrdiff_t)_info.y_stride_seqlen;
    kunlun_ptrdiff_t y_stride_nhead = (kunlun_ptrdiff_t)_info.y_stride_nhead;
    if (_info.data_type == INFINI_DTYPE_F32 && _info.pos_type == INFINI_DTYPE_U32) {
        RoPEF32I32F32(y, x,
                      pos_ids, sin_table, cos_table,
                      seqlen, nhead, dhead,
                      x_stride_seqlen, x_stride_nhead,
                      y_stride_seqlen, y_stride_nhead, reinterpret_cast<kunlunStream_t>(stream));
        return INFINI_STATUS_SUCCESS;
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::rope::kunlun
