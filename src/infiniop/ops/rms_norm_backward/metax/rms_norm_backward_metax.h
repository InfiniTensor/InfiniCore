#ifndef __RMS_NORM_BACKWARD_METAX_H__
#define __RMS_NORM_BACKWARD_METAX_H__

#include "../../../operator.h"
#include "../../../devices/metax/metax_handle.h"
#include "../info.h"
#include <vector>

namespace op::rms_norm_backward::metax {

class Descriptor final : public InfiniopDescriptor {
    RMSNormBackwardInfo _info;
    size_t _workspace_size;

public:
    Descriptor() = delete;
    Descriptor(
        const RMSNormBackwardInfo &info,
        size_t workspace_size,
        infiniDevice_t device,
        int device_id)
        : InfiniopDescriptor{device, device_id},
          _info(info),
          _workspace_size(workspace_size) {}
    
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t grad_x_desc,
        infiniopTensorDescriptor_t grad_w_desc,
        infiniopTensorDescriptor_t grad_y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        float epsilon);

    size_t workspaceSize() const { return _workspace_size; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *grad_x,
        void *grad_w,
        const void *grad_y,
        const void *x,
        const void *w,
        void *stream) const;

private:
    template <typename GradXType, typename AccType, typename WType>
    infiniStatus_t calculate_rms_norm_backward(
        void *workspace,
        void *grad_x_data,
        void *grad_w_data,
        const void *grad_y_data,
        const void *x_data,
        const void *w_data,
        const RMSNormBackwardInfo &info,
        void *stream) const;
};

} // namespace op::rms_norm_backward::metax

#endif // __RMS_NORM_BACKWARD_METAX_H__