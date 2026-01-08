#ifndef __GEMM_ASCEND__
#define __GEMM_ASCEND__
#include "../../../operator.h"

namespace op::gemm::ascend {
class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;
    infiniDtype_t _dtype;
    size_t _workspace_size;

    Descriptor(
        infiniDtype_t dtype,
        size_t workspace_size_,
        Opaque *opaque,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _opaque(opaque),
          _dtype(dtype),
          _workspace_size(workspace_size_) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc);

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *c,
        float beta,
        const void *a,
        const void *b,
        float alpha,
        void *stream) const;
};
} // namespace op::gemm::ascend

#endif
