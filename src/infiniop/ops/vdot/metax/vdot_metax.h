#ifndef __VDOT_METAX_API_H__
#define __VDOT_METAX_API_H__

#include "../../../devices/metax/metax_handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"
#include "../cuda/kernel.cuh"

namespace op::vdot::metax {

class Descriptor final : public InfiniopDescriptor {
  infiniDtype_t _in_dtype;
  infiniDtype_t _out_dtype;
  size_t _length;
  ptrdiff_t _a_stride;
  ptrdiff_t _b_stride;

public:
  Descriptor(infiniDtype_t in_dtype, infiniDtype_t out_dtype, size_t length,
             ptrdiff_t a_stride, ptrdiff_t b_stride, infiniDevice_t device_type,
             int device_id)
      : InfiniopDescriptor{device_type, device_id}, _in_dtype(in_dtype),
        _out_dtype(out_dtype), _length(length), _a_stride(a_stride),
        _b_stride(b_stride) {}

  ~Descriptor();

  size_t workspaceSize() const {
    // Need workspace for FP16/BF16 to accumulate in float
    return (_in_dtype == INFINI_DTYPE_F16 || _in_dtype == INFINI_DTYPE_BF16)
               ? sizeof(float)
               : 0;
  }

  static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,
                               infiniopTensorDescriptor_t out_desc,
                               infiniopTensorDescriptor_t a_desc,
                               infiniopTensorDescriptor_t b_desc);

  infiniStatus_t calculate(void *workspace, size_t workspace_size, void *out,
                           const void *a, const void *b, void *stream) const;
};

} // namespace op::vdot::metax

#endif // __VDOT_METAX_API_H__
