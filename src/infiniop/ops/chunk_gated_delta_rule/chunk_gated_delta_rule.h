// infiniop/ops/chunk_gated_delta_rule.h

#ifndef __INFINIOP_CHUNK_GATED_DELTA_RULE_H__
#define __INFINIOP_CHUNK_GATED_DELTA_RULE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                  \
                                                                               \
    namespace op::chunk_gated_delta_rule::NAMESPACE {                      \
    class Descriptor final : public InfiniopDescriptor {                       \
        struct Opaque;                                                         \
        Opaque *_opaque;                                                       \
        ChunkGatedDeltaRuleInfo _info;                                     \
        size_t _workspace_size;                                                \
                                                                               \
        Descriptor(                                                            \
            Opaque *opaque,                                                    \
            ChunkGatedDeltaRuleInfo info,                                  \
            size_t workspace_size,                                             \
            infiniDevice_t device_type,                                      \
            int device_id)                                                     \
            : InfiniopDescriptor{device_type, device_id},                      \
              _opaque(opaque),                                                 \
              _info(info),                                                     \
              _workspace_size(workspace_size) {}                               \
                                                                               \
    public:                                                                    \
        ~Descriptor();                                                         \
                                                                               \
        size_t workspaceSize() const { return _workspace_size; }               \
                                                                               \
        static infiniStatus_t create(                                          \
            infiniopHandle_t handle,                                           \
            Descriptor **desc_ptr,                                             \
            infiniopTensorDescriptor_t out_desc,                               \
            infiniopTensorDescriptor_t final_state_desc,                       \
            infiniopTensorDescriptor_t q_desc,                                 \
            infiniopTensorDescriptor_t k_desc,                                 \
            infiniopTensorDescriptor_t v_desc,                                 \
            infiniopTensorDescriptor_t g_desc,                                 \
            infiniopTensorDescriptor_t beta_desc,                              \
            infiniopTensorDescriptor_t initial_state_desc,                     \
            bool use_qk_l2norm,                                                \
            size_t chunk_size);                                               \
                                                                               \
        infiniStatus_t calculate(                                              \
            void *workspace, size_t workspace_size,                            \
            void *out, void *final_state,                                      \
            const void *q, const void *k, const void *v,                       \
            const void *g, const void *beta, const void *initial_state,        \
            void *stream) const;                                               \
    };                                                                         \
    }

#endif // __INFINIOP_CHUNK_GATED_DELTA_RULE_H__