#ifndef CAST_NVIDIA_API_H
#define CAST_NVIDIA_API_H

#include "../../../elementwise/nvidia/elementwise_nvidia_api.cuh"
#include <type_traits>

namespace op::cast::nvidia {
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _input_dtype, _output_dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::nvidia::DeviceImpl> _device_info;
    size_t _workspace_size;

    Descriptor(
        infiniDtype_t input_dtype,
        infiniDtype_t output_dtype,
        op::elementwise::ElementwiseInfo info,
        op::elementwise::nvidia::DeviceImpl *device_info,
        size_t workspace_size,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _input_dtype(input_dtype),
          _output_dtype(output_dtype),
          _info(std::move(info)),
          _device_info(std::move(device_info)),
          _workspace_size(workspace_size) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle_,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc);

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};
} // namespace op::cast::nvidia
#endif // CAST_NVIDIA_API_H
