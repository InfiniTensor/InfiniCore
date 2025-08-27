#ifndef __SCATTER_METAX_H__
#define __SCATTER_METAX_H__

#include "../../../operator.h"
#include "../../../devices/metax/metax_handle.h"
#include <vector>

namespace op::scatter::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _input_dtype;
    infiniDtype_t _index_dtype;
    std::vector<size_t> _input_shape;
    std::vector<size_t> _output_shape;
    std::vector<size_t> _index_shape;
    std::vector<size_t> _src_shape;
    std::vector<ptrdiff_t> _input_strides;
    std::vector<ptrdiff_t> _output_strides;
    std::vector<ptrdiff_t> _index_strides;
    std::vector<ptrdiff_t> _src_strides;
    int _dim;

public:
    Descriptor() = default;
    Descriptor(
        infiniDtype_t input_dtype,
        infiniDtype_t index_dtype,
        const std::vector<size_t> &input_shape,
        const std::vector<size_t> &output_shape,
        const std::vector<size_t> &index_shape,
        const std::vector<size_t> &src_shape,
        const std::vector<ptrdiff_t> &input_strides,
        const std::vector<ptrdiff_t> &output_strides,
        const std::vector<ptrdiff_t> &index_strides,
        const std::vector<ptrdiff_t> &src_strides,
        int dim,
        infiniDevice_t device,
        int device_id)
        : InfiniopDescriptor{device, device_id},
          _input_dtype(input_dtype),
          _index_dtype(index_dtype),
          _input_shape(input_shape),
          _output_shape(output_shape),
          _index_shape(index_shape),
          _src_shape(src_shape),
          _input_strides(input_strides),
          _output_strides(output_strides),
          _index_strides(index_strides),
          _src_strides(src_strides),
          _dim(dim) {}
    
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t index_desc,
        infiniopTensorDescriptor_t src_desc,
        int dim);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        const void *input,
        const void *index,
        const void *src,
        void *stream) const;

private:
    template <typename T, typename IndexT>
    infiniStatus_t scatterMetax(
        void *output_data,
        const void *input_data,
        const void *index_data,
        void *stream) const;
};

} // namespace op::scatter::metax

#endif // __SCATTER_METAX_H__