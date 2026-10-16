#ifndef __W4A8_GROUP_GEMM_H__
#define __W4A8_GROUP_GEMM_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::w4a8_group_gemm::NAMESPACE {                   \
    class Descriptor final : public InfiniopDescriptor {         \
        size_t _m;                                               \
        size_t _n;                                               \
        size_t _k;                                               \
        size_t _experts;                                         \
        infiniDtype_t _out_dtype;                                \
        bool _trans_weight;                                      \
        bool _has_sorted_token_ids;                              \
        bool _has_bias;                                          \
                                                                 \
        Descriptor(size_t m,                                     \
                   size_t n,                                     \
                   size_t k,                                     \
                   size_t experts,                               \
                   infiniDtype_t out_dtype,                      \
                   bool trans_weight,                            \
                   bool has_sorted_token_ids,                    \
                   bool has_bias,                                \
                   infiniDevice_t device_type,                   \
                   int device_id)                                \
            : InfiniopDescriptor{device_type, device_id},        \
              _m(m),                                             \
              _n(n),                                             \
              _k(k),                                             \
              _experts(experts),                                 \
              _out_dtype(out_dtype),                             \
              _trans_weight(trans_weight),                       \
              _has_sorted_token_ids(has_sorted_token_ids),       \
              _has_bias(has_bias) {}                             \
                                                                 \
    public:                                                      \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t out_desc,                 \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t weight_desc,              \
            infiniopTensorDescriptor_t input_scale_desc,         \
            infiniopTensorDescriptor_t weight_scale_desc,        \
            infiniopTensorDescriptor_t tokens_per_experts_desc,  \
            infiniopTensorDescriptor_t sorted_token_ids_desc,    \
            infiniopTensorDescriptor_t bias_desc,                \
            bool trans_weight);                                  \
                                                                 \
        infiniStatus_t calculate(void *out,                      \
                                 const void *input,              \
                                 const void *weight,             \
                                 const void *input_scale,        \
                                 const void *weight_scale,       \
                                 const void *tokens_per_experts, \
                                 const void *sorted_token_ids,   \
                                 const void *bias,               \
                                 void *stream) const;            \
    };                                                           \
    }

#endif
