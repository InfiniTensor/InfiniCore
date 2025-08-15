#ifndef __CROSS_ENTROPY_LOSS_BACKWARD_METAX_API_H__
#define __CROSS_ENTROPY_LOSS_BACKWARD_METAX_API_H__

#include "../cross_entropy_loss_backward.h"
#include "../../../elementwise/metax/elementwise_metax_api.h"

CROSS_ENTROPY_LOSS_BACKWARD_ELEMENTWISE_DESCRIPTOR(cross_entropy_loss_backward, metax)


#define CREATE_CROSS_ENTROPY_LOSS_BACKWARD_ELEMENTWISE_METAX_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC, BATCH_SIZE) \
                                                                                              \
    auto info_result = op::elementwise::ElementwiseInfo::create(OUT_DESC, INPUT_DESC_VEC);    \
    CHECK_RESULT(info_result);                                                                \
    auto info = info_result.take();                                                           \
    auto workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void *);       \
                                                                                              \
    auto device_impl_result = op::elementwise::metax::DeviceImpl::create(HANDLE->internal()); \
    CHECK_RESULT(device_impl_result);                                                         \
                                                                                              \
    *desc_ptr = new Descriptor(                                                               \
        DTYPE,                                                                                \
        std::move(info),                                                                      \
        std::move(device_impl_result.take()),                                                 \
        workspace_size,                                                                       \
        HANDLE->device,                                                                       \
        HANDLE->device_id,                                                                    \
        BATCH_SIZE);

#endif // __CROSS_ENTROPY_LOSS_BACKWARD_METAX_API_H__
