#pragma once
#include "../../../operator.h"
#include "infiniop/ops/apply_rope.h"
#include <memory>
namespace op::apply_rope::ascend {
class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    std::unique_ptr<Opaque> opaque_;
    size_t workspace_size_;
    Descriptor(std::unique_ptr<Opaque> opaque, size_t bytes,
               infiniDevice_t device, int device_id);

public:
    ~Descriptor();
    size_t workspaceSize() const { return workspace_size_; }
    static infiniStatus_t create(
        infiniopHandle_t handle, Descriptor **desc,
        infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t k,
        infiniopTensorDescriptor_t cos, infiniopTensorDescriptor_t sin);
    infiniStatus_t calculate(
        void *workspace, size_t workspace_size, void *q, void *k,
        const void *cos, const void *sin, void *stream) const;
};
} // namespace op::apply_rope::ascend
