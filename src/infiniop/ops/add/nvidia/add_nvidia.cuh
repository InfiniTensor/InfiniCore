#ifndef __ADD_CUDA_API_H__
#define __ADD_CUDA_API_H__

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../elementwise/nvidia/elementwise_nvidia_api.cuh"

namespace op::add::nvidia {
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::shared_ptr<device::nvidia::Handle::Internal> _internal;
    size_t _workspace_size;

    Descriptor(
        infiniDtype_t dtype,
        op::elementwise::ElementwiseInfo info,
        std::shared_ptr<device::nvidia::Handle::Internal> internal,
        size_t workspace_size,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)),
          _internal(std::move(internal)),
          _workspace_size(workspace_size) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        const void *a,
        const void *b,
        void *stream) const;
};
} // namespace op::add::nvidia

#endif // __ADD_CUDA_API_H__
