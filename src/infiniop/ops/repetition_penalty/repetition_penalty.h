#ifndef __REPETITION_PENALTY_H__
#define __REPETITION_PENALTY_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::repetition_penalty::NAMESPACE {         \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
                                                          \
        RepetitionPenaltyInfo _info;                      \
        size_t _workspace_size;                           \
                                                          \
        Descriptor(                                       \
            RepetitionPenaltyInfo info,                   \
            size_t workspace_size,                        \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _info(info),                                \
              _workspace_size(workspace_size) {}          \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        size_t workspaceSize() const { return _workspace_size; } \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t logits_desc,       \
            infiniopTensorDescriptor_t mask_desc);        \
                                                          \
        infiniStatus_t calculate(                         \
            void *workspace,                              \
            size_t workspace_size,                        \
            void *logits,                                 \
            const void *mask,                             \
            const float *repetition_penalties,            \
            void *stream) const;                          \
    };                                                    \
    }

#endif // __REPETITION_PENALTY_H__
