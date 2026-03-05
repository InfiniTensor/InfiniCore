#ifndef __PRELU_NVIDIA_H__
#define __PRELU_NVIDIA_H__

#include "../../../operator.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include <vector>

namespace op::prelu::nvidia {

enum class WeightMode : int {
    SCALAR = 0,
    PER_CHANNEL = 1,
    ELEMENTWISE = 2,
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t numel;
    size_t ndim;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> x_strides;
    std::vector<size_t> weight_shape;
    std::vector<ptrdiff_t> weight_strides;
    WeightMode weight_mode;
    ptrdiff_t weight_stride0;
    int channel_axis;

    Descriptor(
        infiniDtype_t dtype,
        size_t numel,
        size_t ndim,
        std::vector<size_t> shape,
        std::vector<ptrdiff_t> y_strides,
        std::vector<ptrdiff_t> x_strides,
        std::vector<size_t> weight_shape,
        std::vector<ptrdiff_t> weight_strides,
        WeightMode weight_mode,
        ptrdiff_t weight_stride0,
        int channel_axis,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          numel(numel),
          ndim(ndim),
          shape(std::move(shape)),
          y_strides(std::move(y_strides)),
          x_strides(std::move(x_strides)),
          weight_shape(std::move(weight_shape)),
          weight_strides(std::move(weight_strides)),
          weight_mode(weight_mode),
          weight_stride0(weight_stride0),
          channel_axis(channel_axis) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        std::vector<const void *> inputs,
        void *stream) const;
};

} // namespace op::prelu::nvidia

#endif // __PRELU_NVIDIA_H__

