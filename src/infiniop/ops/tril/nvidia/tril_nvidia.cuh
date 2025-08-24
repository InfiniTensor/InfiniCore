#ifndef __TRIL_NVIDIA_CUH__
#define __TRIL_NVIDIA_CUH__

#include "../../../operator.h"
#include "../../../devices/nvidia/nvidia_handle.h"

namespace op::tril::nvidia {

class Descriptor : public InfiniopDescriptor {
public:
    Descriptor() = default;
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t output_desc,
        int diagonal);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        void *input,
        void *stream) const;

    infiniStatus_t calculateInplace(
        void *workspace,
        size_t workspace_size,
        void *input_output,
        void *stream) const;

private:
    infiniopTensorDescriptor_t _input_desc;
    infiniopTensorDescriptor_t _output_desc;
    int _diagonal;
    device::nvidia::Handle *_handle;
};

} // namespace op::tril::nvidia

#endif // __TRIL_NVIDIA_CUH__