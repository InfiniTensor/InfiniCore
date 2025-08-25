#ifndef __SCATTER_CPU_H__
#define __SCATTER_CPU_H__

#include "../../../operator.h"
#include "../../../devices/cpu/cpu_handle.h"
#include <vector>

namespace op::scatter::cpu {

class Descriptor : public InfiniopDescriptor {
public:
    Descriptor() = default;
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
    infiniopTensorDescriptor_t _input_desc;
    infiniopTensorDescriptor_t _output_desc;
    infiniopTensorDescriptor_t _index_desc;
    infiniopTensorDescriptor_t _src_desc;
    int _dim;
    std::vector<size_t> _input_shape;
    std::vector<size_t> _output_shape;
    std::vector<size_t> _src_shape;
    std::vector<ptrdiff_t> _input_strides;
    std::vector<ptrdiff_t> _output_strides;
    std::vector<ptrdiff_t> _index_strides;
    std::vector<ptrdiff_t> _src_strides;
    infiniDtype_t _dtype;
    infiniDtype_t _index_dtype;
    device::cpu::Handle *_handle;
};

} // namespace op::scatter::cpu

#endif // __SCATTER_CPU_H__