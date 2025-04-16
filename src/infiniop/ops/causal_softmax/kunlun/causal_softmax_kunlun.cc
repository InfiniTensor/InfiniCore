#include "causal_softmax_kunlun.h"
#include "../../../devices/kunlun/kunlun_handle.h"
#include <memory>
#include <stdint.h>

void causalSoftmaxF32(void *y, const void *x, void *workspace, int batch, int seq_len, int total_seq_len,
                      int y_stride_b, int y_stride_i, int x_stride_b, int x_stride_i, XPUStream stream);

namespace op::causal_softmax::kunlun {

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
    infiniopTensorDescriptor_t x_desc) {
    auto info = CausalSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(info);
    auto info_out = info.take();
    auto workspace_size = info_out.total_seq_len * info_out.seq_len * infiniSizeOf(info_out.dtype);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::kunlun::Handle *>(handle)->internal()},
        info.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(void *y, const void *x, void *workspace, infiniDtype_t dtype,
                            size_t batch_size, size_t seq_len, size_t total_seq_len,
                            ptrdiff_t y_stride_b, ptrdiff_t y_stride_i,
                            ptrdiff_t x_stride_b, ptrdiff_t x_stride_i,
                            XPUStream stream) {
    if (dtype == INFINI_DTYPE_F32) {
        causalSoftmaxF32(y, x, workspace, (int)batch_size, (int)seq_len, (int)total_seq_len,
                         (int)y_stride_b, (int)y_stride_i, (int)x_stride_b,
                         (int)x_stride_i, stream);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     void *stream_) const {
    kunlunStream_t stream = (kunlunStream_t)stream_;
    CHECK_STATUS(launchKernel(y, x, workspace, _info.dtype, _info.batch_size, _info.seq_len,
                              _info.total_seq_len, _info.y_stride_b, _info.y_stride_i,
                              _info.x_stride_b, _info.x_stride_i, stream));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::kunlun
