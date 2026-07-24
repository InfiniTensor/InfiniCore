#ifndef __INFINIOP_KIMI_DELTA_ATTENTION_H__
#define __INFINIOP_KIMI_DELTA_ATTENTION_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                    \
                                                                                 \
    namespace op::kimi_delta_attention::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {                         \
        struct Opaque;                                                           \
        Opaque *_opaque;                                                         \
        KimiDeltaAttentionInfo _info;                                            \
        size_t _workspace_size;                                                  \
                                                                                 \
        Descriptor(Opaque *opaque,                                               \
                   KimiDeltaAttentionInfo info,                                  \
                   size_t workspace_size,                                        \
                   infiniDevice_t device_type,                                   \
                   int device_id)                                                \
            : InfiniopDescriptor{device_type, device_id},                        \
              _opaque(opaque),                                                   \
              _info(info),                                                       \
              _workspace_size(workspace_size) {}                                 \
                                                                                 \
    public:                                                                      \
        ~Descriptor();                                                           \
                                                                                 \
        size_t workspaceSize() const { return _workspace_size; }                 \
                                                                                 \
        static infiniStatus_t create(                                            \
            infiniopHandle_t handle,                                             \
            Descriptor **desc_ptr,                                               \
            infiniopTensorDescriptor_t out_desc,                                 \
            infiniopTensorDescriptor_t initial_state_desc,                       \
            infiniopTensorDescriptor_t final_state_desc,                         \
            infiniopTensorDescriptor_t q_desc,                                   \
            infiniopTensorDescriptor_t k_desc,                                   \
            infiniopTensorDescriptor_t v_desc,                                   \
            infiniopTensorDescriptor_t g_desc,                                   \
            infiniopTensorDescriptor_t beta_desc,                                \
            infiniopTensorDescriptor_t A_log_desc,                               \
            infiniopTensorDescriptor_t dt_bias_desc,                             \
            infiniopTensorDescriptor_t cu_seqlens_desc,                          \
            infiniopTensorDescriptor_t initial_state_indices_desc,               \
            infiniopTensorDescriptor_t final_state_indices_desc,                 \
            float scale,                                                         \
            float lower_bound,                                                   \
            bool use_qk_l2norm);                                                 \
                                                                                 \
        infiniStatus_t calculate(                                                \
            void *workspace, size_t workspace_size,                              \
            void *out, void *initial_state, void *final_state,                   \
            const void *q, const void *k, const void *v,                         \
            const void *g, const void *beta, const void *A_log,                  \
            const void *dt_bias, const void *cu_seqlens,                         \
            const void *initial_state_indices,                                   \
            const void *final_state_indices,                                     \
            void *stream) const;                                                 \
    };                                                                           \
    }

#endif
