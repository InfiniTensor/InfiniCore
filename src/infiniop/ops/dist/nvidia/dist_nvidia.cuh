#ifndef __DIST_NVIDIA_H__
#define __DIST_NVIDIA_H__

#include "../../../operator.h"
#include "../../../devices/nvidia/nvidia_common.cuh"

namespace op::dist::nvidia {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t _input_size;
    double _p;
    ptrdiff_t _x1_stride;
    ptrdiff_t _x2_stride;

    Descriptor(infiniDtype_t dtype, size_t input_size, double p,
               ptrdiff_t x1_stride, ptrdiff_t x2_stride,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _input_size(input_size),
          _p(p),
          _x1_stride(x1_stride),
          _x2_stride(x2_stride) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x1_desc,
        infiniopTensorDescriptor_t x2_desc,
        double p);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x1,
        const void *x2,
        void *stream) const;
};

} // namespace op::dist::nvidia

#endif // __DIST_NVIDIA_H__
