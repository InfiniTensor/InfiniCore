#ifndef __EQUAL_NVIDIA_CUH__
#define __EQUAL_NVIDIA_CUH__

#include "../../../operator.h"
#include "../../../handle.h"
#include "../../../tensor.h"

namespace op::equal::nvidia {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    std::vector<size_t> _shape;
    std::vector<ptrdiff_t> _a_strides;
    std::vector<ptrdiff_t> _b_strides;

public:
    Descriptor() = default;
    Descriptor(infiniDtype_t dtype, 
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &a_strides,
               const std::vector<ptrdiff_t> &b_strides,
               infiniDevice_t device,
               int device_id)
        : InfiniopDescriptor{device, device_id},
          _dtype(dtype), _shape(shape), _a_strides(a_strides), 
          _b_strides(b_strides) {}
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;

private:
    template <typename T>
    infiniStatus_t compareArraysCuda(
        const void *a_data,
        const void *b_data,
        size_t total_elements,
        const std::vector<ptrdiff_t> &a_strides,
        const std::vector<ptrdiff_t> &b_strides,
        bool *result,
        void *stream) const;
};

} // namespace op::equal::nvidia

#endif // __EQUAL_NVIDIA_CUH__