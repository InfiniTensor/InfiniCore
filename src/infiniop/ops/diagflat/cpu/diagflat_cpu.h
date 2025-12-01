#ifndef __DIAGFLAT_CPU_H__
#define __DIAGFLAT_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <vector>

namespace op::diagflat::cpu {
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    std::vector<size_t> _input_shape;
    std::vector<ptrdiff_t> _input_strides;
    std::vector<size_t> _output_shape;
    int64_t _offset;
    size_t _workspace_size;
    bool _input_contiguous;
    size_t _ndim;

    Descriptor(
        infiniDtype_t dtype,
        std::vector<size_t> input_shape,
        std::vector<ptrdiff_t> input_strides,
        std::vector<size_t> output_shape,
        int64_t offset,
        size_t workspace_size,
        bool input_contiguous,
        size_t ndim,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _input_shape(std::move(input_shape)),
          _input_strides(std::move(input_strides)),
          _output_shape(std::move(output_shape)),
          _offset(offset),
          _workspace_size(workspace_size),
          _input_contiguous(input_contiguous),
          _ndim(ndim) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs,
        int64_t offset);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};

} // namespace op::diagflat::cpu

#endif // __DIAGFLAT_CPU_H__