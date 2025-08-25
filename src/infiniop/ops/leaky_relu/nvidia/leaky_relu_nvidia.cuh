#ifndef LEAKY_RELU_NVIDIA_API_H
#define LEAKY_RELU_NVIDIA_API_H

#include "../../../../utils/custom_types.h"
#include "../../../elementwise/nvidia/elementwise_nvidia_api.cuh"

namespace op::leaky_relu::nvidia {
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::nvidia::DeviceImpl> _device_info;
    size_t _workspace_size;
    float _negative_slope;

    Descriptor(
        infiniDtype_t dtype,
        op::elementwise::ElementwiseInfo info,
        op::elementwise::nvidia::DeviceImpl *device_info,
        size_t workspace_size,
        infiniDevice_t device_type,
        int device_id,
        float negative_slope)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)),
          _device_info(std::move(device_info)),
          _workspace_size(workspace_size),
          _negative_slope(negative_slope) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc,
        float negative_slope);

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};
} // namespace op::leaky_relu::nvidia
#endif // LEAKY_RELU_NVIDIA_API_H