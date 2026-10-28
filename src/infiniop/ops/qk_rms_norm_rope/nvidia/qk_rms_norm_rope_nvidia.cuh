#pragma once
#include "../../../operator.h"
#include "infiniop/ops/qk_rms_norm_rope.h"
namespace op::qk_rms_norm_rope::nvidia {
struct Info {
    size_t tokens, q_heads, k_heads, max_positions;
    ptrdiff_t q_stride, k_stride, qo_stride, ko_stride, cos_stride, sin_stride, pos_stride;
    float epsilon;
};
class Descriptor final : public InfiniopDescriptor {
    Info info_;
    Descriptor(Info info, infiniDevice_t type, int id) : InfiniopDescriptor{type, id}, info_(info) {}

public:
    size_t workspaceSize() const { return 0; }
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                 infiniopTensorDescriptor_t q_out,
                                 infiniopTensorDescriptor_t k_out,
                                 infiniopTensorDescriptor_t q,
                                 infiniopTensorDescriptor_t k,
                                 infiniopTensorDescriptor_t q_weight,
                                 infiniopTensorDescriptor_t k_weight,
                                 infiniopTensorDescriptor_t cos,
                                 infiniopTensorDescriptor_t sin,
                                 infiniopTensorDescriptor_t positions, float epsilon);
    infiniStatus_t calculate(void *workspace, size_t workspace_size,
                             void *q_out,
                             void *k_out,
                             const void *q,
                             const void *k,
                             const void *q_weight,
                             const void *k_weight,
                             const void *cos,
                             const void *sin,
                             const void *positions, void *stream) const;
};
} // namespace op::qk_rms_norm_rope::nvidia
