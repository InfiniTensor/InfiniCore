#include "softmax_bang.h"
#include "../../../devices/bang/common_bang.h"

namespace op::softmax::bang {

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlTensorDescriptor_t x_desc = nullptr;
    cnnlTensorDescriptor_t y_desc = nullptr;

    ~Opaque() {
        if (x_desc) {
            cnnlDestroyTensorDescriptor(x_desc);
        }
        if (y_desc) {
            cnnlDestroyTensorDescriptor(y_desc);
        }
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

static infiniStatus_t setSoftmaxTensor(cnnlTensorDescriptor_t desc, const SoftmaxInfo &info) {
    int dims[3] = {
        static_cast<int>(info.othersize / info.stride),
        static_cast<int>(info.dimsize),
        static_cast<int>(info.stride),
    };
    CHECK_BANG(cnnlSetTensorDescriptor(
        desc,
        CNNL_LAYOUT_ARRAY,
        device::bang::getCnnlDtype(info.dtype),
        3,
        dims));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis) {
    auto result = SoftmaxInfo::create(y_desc, x_desc, axis);
    CHECK_RESULT(result);
    auto info = result.take();

    cnnlTensorDescriptor_t cnnl_x = nullptr;
    cnnlTensorDescriptor_t cnnl_y = nullptr;
    CHECK_BANG(cnnlCreateTensorDescriptor(&cnnl_x));
    CHECK_BANG(cnnlCreateTensorDescriptor(&cnnl_y));
    CHECK_STATUS(setSoftmaxTensor(cnnl_x, info));
    CHECK_STATUS(setSoftmaxTensor(cnnl_y, info));

    auto handle_bang = reinterpret_cast<device::bang::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{handle_bang->internal(), cnnl_x, cnnl_y},
        std::move(info),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    CHECK_STATUS(_opaque->internal->useCnnl(
        queue,
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlSoftmaxForward(
                handle,
                CNNL_SOFTMAX_ACCURATE,
                CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION,
                nullptr,
                _opaque->x_desc,
                x,
                nullptr,
                _opaque->y_desc,
                y));
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync(queue);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::softmax::bang
