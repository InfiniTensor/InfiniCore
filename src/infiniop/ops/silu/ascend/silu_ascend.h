#ifndef __SILU_ASCEND__
#define __SILU_ASCEND__
#include "../../../operator.h"
#include <array>

namespace op::silu::ascend {
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

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        std::array<infiniopTensorDescriptor_t, 1> ab_desc);

    size_t workspaceSize() const { return _workspace_size; };

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        std::array<const void *, 1> ab,
        void *stream) const;
};
} // namespace op::silu::ascend

#endif
