#ifndef __TRIL_METAX_H__
#define __TRIL_METAX_H__

#include "../../../operator.h"
#include "../../../handle.h"
#include "../../../tensor.h"

namespace op::tril::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    std::vector<size_t> _shape;
    std::vector<ptrdiff_t> _input_strides;
    std::vector<ptrdiff_t> _output_strides;
    int _diagonal;

public:
    Descriptor() = default;
    Descriptor(infiniDtype_t dtype,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &input_strides,
               const std::vector<ptrdiff_t> &output_strides,
               int diagonal,
               infiniDevice_t device,
               int device_id)
        : InfiniopDescriptor{device, device_id},
          _dtype(dtype), _shape(shape),
          _input_strides(input_strides), _output_strides(output_strides),
          _diagonal(diagonal) {}
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
        const void *input,
        void *stream) const;

private:
    template <typename T>
    infiniStatus_t trilMetax(
        void *output_data,
        const void *input_data,
        void *stream) const;
};

} // namespace op::tril::metax

#endif // __TRIL_METAX_H__