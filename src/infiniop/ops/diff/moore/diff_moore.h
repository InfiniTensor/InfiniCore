#ifndef __DIFF_MOORE_H__
#define __DIFF_MOORE_H__

#include "../../../operator.h"
#include "../../../devices/moore/moore_common.h"

namespace op::diff::moore {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t _ndim;
    int _dim;
    int _n;
    std::vector<size_t> _input_shape;
    std::vector<size_t> _output_shape;
    size_t _input_size;
    size_t _output_size;

    Descriptor(infiniDtype_t dtype, size_t ndim, int dim, int n,
               std::vector<size_t> input_shape, std::vector<size_t> output_shape,
               size_t input_size, size_t output_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _ndim(ndim),
          _dim(dim),
          _n(n),
          _input_shape(std::move(input_shape)),
          _output_shape(std::move(output_shape)),
          _input_size(input_size),
          _output_size(output_size) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        int dim,
        int n);

    size_t workspaceSize() const { return _input_size * sizeof(float); }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::diff::moore

#endif // __DIFF_MOORE_H__
