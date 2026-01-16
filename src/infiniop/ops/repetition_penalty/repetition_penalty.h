#ifndef __REPETITION_PENALTY_H__
#define __REPETITION_PENALTY_H__

#include "../../operator.h"
#include "info.h"
#include <stdint.h>

#define DESCRIPTOR(NAMESPACE)                                     \
                                                                    \
    namespace op::repetition_penalty::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {            \
        struct Opaque;                                              \
        Opaque *_opaque;                                            \
                                                                    \
        RepetitionPenaltyInfo _info;                               \
        size_t _min_workspace_size;                                 \
                                                                    \
        Descriptor(                                                 \
            RepetitionPenaltyInfo info,                             \
            size_t min_workspace_size,                              \
            Opaque *opaque,                                         \
            infiniDevice_t device_type,                              \
            int device_id)                                          \
            : InfiniopDescriptor{device_type, device_id},           \
              _opaque(opaque),                                      \
              _info(info),                                          \
              _min_workspace_size(min_workspace_size) {}            \
                                                                    \
    public:                                                         \
        ~Descriptor();                                              \
                                                                    \
        static infiniStatus_t create(                               \
            infiniopHandle_t handle,                                \
            Descriptor **desc_ptr,                                  \
            infiniopTensorDescriptor_t logits_desc);                \
                                                                    \
        size_t minWorkspaceSize() const;                            \
                                                                    \
        infiniStatus_t calculate(                                   \
            void *workspace,                                        \
            size_t workspace_size,                                  \
            void *logits,                                           \
            const float *repetition_penalties,                      \
            const uint32_t *token_indices,                          \
            const size_t *token_offsets,                            \
            size_t total_indices,                                   \
            void *stream) const;                                    \
    };                                                              \
    }

#endif // __REPETITION_PENALTY_H__
