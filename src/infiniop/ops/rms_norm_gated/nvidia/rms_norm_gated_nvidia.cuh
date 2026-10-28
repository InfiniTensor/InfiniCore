#pragma once
#include "../../../operator.h"
#include "infiniop/ops/rms_norm_gated.h"
namespace op::rms_norm_gated::nvidia {
struct Info {
    size_t tokens, heads;
    ptrdiff_t xs, zs, ys;
    float epsilon;
};
class Descriptor final : public InfiniopDescriptor {
    Info info_;
    Descriptor(Info info, infiniDevice_t type, int id) : InfiniopDescriptor{type, id}, info_(info) {}

public:
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t out, infiniopTensorDescriptor_t x, infiniopTensorDescriptor_t gate, infiniopTensorDescriptor_t weight, float epsilon);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *out, const void *x, const void *gate, const void *weight, void *stream) const;
};
} // namespace op::rms_norm_gated::nvidia
