#include "swiglu_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"

namespace op::swiglu::ascend {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t c_desc,
                                  infiniopTensorDescriptor_t a_desc,
                                  infiniopTensorDescriptor_t b_desc) {
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    auto dtype = c_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

    SwigluInfo info;
    CHECK_STATUS(createSwigluInfo(info, c_desc, a_desc, b_desc));

    *desc_ptr = new Descriptor(std::move(info), handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

extern "C" infiniStatus_t swiglu_kernel_launch(void *c, void *a, void *b,
                                               int dtype, int batch, int seq, int hd,
                                               int stride_batch_c, int stride_batch_a, int stride_batch_b,
                                               int stride_seq_c, int stride_seq_a, int stride_seq_b, void *stream);

infiniStatus_t Descriptor::calculate(void *c,
                                     const void *a,
                                     const void *b,
                                     void *stream) const {
    std::cout << "info.dtype: " << _info.dtype << std::endl;
    int batch = _info.ndim == 2 ? 1 : _info.shape[0];
    int seq_len = _info.ndim == 2 ? _info.shape[0] : _info.shape[1];
    int hidden_size = _info.shape[_info.ndim - 1];
    int stride_batch_c = _info.ndim == 2 ? 1 : _info.c_strides[0];
    int stride_batch_a = _info.ndim == 2 ? 1 : _info.a_strides[0];
    int stride_batch_b = _info.ndim == 2 ? 1 : _info.b_strides[0];
    int stride_seq_c = _info.ndim == 2 ? _info.c_strides[0] : _info.c_strides[1];
    int stride_seq_a = _info.ndim == 2 ? _info.a_strides[0] : _info.a_strides[1];
    int stride_seq_b = _info.ndim == 2 ? _info.b_strides[0] : _info.b_strides[1];
    auto status = swiglu_kernel_launch(c, (void *)a, (void *)b, _info.dtype, batch, seq_len, hidden_size, stride_batch_c, stride_batch_a, stride_batch_b, stride_seq_c, stride_seq_a, stride_seq_b, stream);
    // CHECK_ACL(aclrtSynchronizeStream((aclrtStream)stream));
    return status;
}

} // namespace op::swiglu::ascend