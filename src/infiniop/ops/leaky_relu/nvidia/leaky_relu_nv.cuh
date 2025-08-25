#ifndef __LEAKY_RELU_NV_CUH__
#define __LEAKY_RELU_NV_CUH__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

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
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)),
          _device_info(std::move(device_info)),
          _workspace_size(workspace_size),
          _negative_slope(0.01f) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs,
        float negative_slope);

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
        
    friend void setDescriptorNegativeSlope(Descriptor* desc, float slope);
};

}

#endif // __LEAKY_RELU_NV_CUH__