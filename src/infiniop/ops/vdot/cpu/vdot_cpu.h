#ifndef __VDOT_CPU_H__
#define __VDOT_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::vdot::cpu {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _in_dtype;
    infiniDtype_t _out_dtype;
    size_t _length;
    ptrdiff_t _a_stride;
    ptrdiff_t _b_stride;

public:
    Descriptor(infiniDtype_t in_dtype,
               infiniDtype_t out_dtype,
               size_t length,
               ptrdiff_t a_stride,
               ptrdiff_t b_stride,
               infiniDevice_t device_type,
               int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _in_dtype(in_dtype),
          _out_dtype(out_dtype),
          _length(length),
          _a_stride(a_stride),
          _b_stride(b_stride) {}

    ~Descriptor();

    size_t workspaceSize() const { return 0; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *out,
        const void *a,
        const void *b,
        void *stream) const;
};

} // namespace op::vdot::cpu

#endif // __VDOT_CPU_H__
