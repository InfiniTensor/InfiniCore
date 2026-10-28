#pragma once
#include "../../../operator.h"
#include "infiniop/ops/vision_qkv_rope_pad.h"
namespace op::vision_qkv_rope_pad::nvidia {
struct Info {
    size_t tokens, max_positions;
    ptrdiff_t qs, qp, qh, cs, ss, rs, ps;
    ptrdiff_t qos, qoh, kos, koh, vos, voh;
};
class Descriptor final : public InfiniopDescriptor {
    Info info_;
    Descriptor(Info info, infiniDevice_t type, int id) : InfiniopDescriptor{type, id}, info_(info) {}

public:
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_out, infiniopTensorDescriptor_t k_out, infiniopTensorDescriptor_t v_out, infiniopTensorDescriptor_t qkv, infiniopTensorDescriptor_t cos, infiniopTensorDescriptor_t sin, infiniopTensorDescriptor_t rows, infiniopTensorDescriptor_t cols);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *q_out, void *k_out, void *v_out, const void *qkv, const void *cos, const void *sin, const void *rows, const void *cols, void *stream) const;
};
} // namespace op::vision_qkv_rope_pad::nvidia
