#ifndef __SIMPLE_GLA_ATTENTION_CPU_H__
#define __SIMPLE_GLA_ATTENTION_CPU_H__

#include "../../../operator.h"

namespace op::simple_gla_attention::cpu {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _q_dtype{INFINI_DTYPE_INVALID};
    size_t _B{};
    size_t _T{};
    size_t _H{};
    size_t _D{};

    Descriptor(infiniDtype_t q_dtype, size_t B, size_t T, size_t H, size_t D, infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _q_dtype(q_dtype),
          _B(B),
          _T(T),
          _H(H),
          _D(D) {}

public:
    ~Descriptor() = default;

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t g_gamma_desc);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *out,
        void const *q,
        void const *k,
        void const *v,
        void const *g_gamma,
        float scale,
        void *stream) const;
};

} // namespace op::simple_gla_attention::cpu

#endif
