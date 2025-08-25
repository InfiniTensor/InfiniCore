#ifndef __INDEX_COPY_INPLACE_NVIDIA_CUH__
#define __INDEX_COPY_INPLACE_NVIDIA_CUH__

#include "../../../operator.h"
#include "../../../devices/nvidia/nvidia_handle.h"

namespace op::index_copy_inplace::nvidia {

class Descriptor : public InfiniopDescriptor {
public:
    Descriptor() = default;
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t output_desc,
        int dim,
        infiniopTensorDescriptor_t index_desc);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *input,
        void *output,
        const void *index,
        void *stream) const;

private:
    infiniopTensorDescriptor_t _input_desc;
    infiniopTensorDescriptor_t _output_desc;
    infiniopTensorDescriptor_t _index_desc;
    int _dim;
    device::nvidia::Handle *_handle;
};

} // namespace op::index_copy_inplace::nvidia

#endif // __INDEX_COPY_INPLACE_NVIDIA_CUH__