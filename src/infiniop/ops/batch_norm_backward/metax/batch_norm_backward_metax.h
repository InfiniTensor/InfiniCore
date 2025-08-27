#ifndef __BATCH_NORM_BACKWARD_METAX_H__
#define __BATCH_NORM_BACKWARD_METAX_H__

#include "../../../operator.h"
#include "../../../devices/metax/metax_handle.h"
#include "../batch_norm_backward.h"
#include <vector>

namespace op::batch_norm_backward::metax {

class Descriptor final : public InfiniopDescriptor {
    BatchNormBackwardInfo _info;
    size_t _workspace_size;

public:
    Descriptor() = delete;
    Descriptor(
        const BatchNormBackwardInfo &info,
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
        infiniopTensorDescriptor_t grad_input_desc,
        infiniopTensorDescriptor_t grad_weight_desc,
        infiniopTensorDescriptor_t grad_bias_desc,
        infiniopTensorDescriptor_t grad_output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t running_mean_desc,
        infiniopTensorDescriptor_t running_var_desc,
        float momentum,
        float eps);

    size_t workspaceSize() const { return _workspace_size; }
    
    infiniStatus_t get_workspace_size(size_t *size) const {
        if (!size) {
            return INFINI_STATUS_BAD_PARAM;
        }
        *size = _workspace_size;
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *grad_input,
        void *grad_weight,
        void *grad_bias,
        const void *grad_output,
        const void *input,
        const void *weight,
        const void *running_mean,
        const void *running_var,
        void *stream) const;

private:
    template <typename T>
    infiniStatus_t calculate_batch_norm_backward(
        void *workspace,
        void *grad_input_data,
        void *grad_weight_data,
        void *grad_bias_data,
        const void *grad_output_data,
        const void *input_data,
        const void *weight_data,
        const void *running_mean_data,
        const void *running_var_data,
        const BatchNormBackwardInfo &info,
        void *stream) const;
};

} // namespace op::batch_norm_backward::metax

#endif // __BATCH_NORM_BACKWARD_METAX_H__