#pragma once
#include "../../../operator.h"
#include "infiniop/ops/gdn_qkv_norm.h"
namespace op::gdn_qkv_norm::nvidia {
struct Info {
    size_t tokens;
    ptrdiff_t qs, qh, ks, kh, vs, vh;
};
class Descriptor final : public InfiniopDescriptor {
    Info info_;
    Descriptor(Info info, infiniDevice_t type, int id) : InfiniopDescriptor{type, id}, info_(info) {}

public:
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_out, infiniopTensorDescriptor_t k_out, infiniopTensorDescriptor_t v_out, infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t k, infiniopTensorDescriptor_t v);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *q_out, void *k_out, void *v_out, const void *q, const void *k, const void *v, void *stream) const;
};
} // namespace op::gdn_qkv_norm::nvidia
