#ifndef GPTQ_MARLIN_REPACK_H
#define GPTQ_MARLIN_REPACK_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::gptq_marlin_repack::NAMESPACE {                \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        GptqMarlinRepackInfo _info;                              \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            GptqMarlinRepackInfo info,                           \
            size_t workspace_size,                               \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t output_desc,              \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t perm_desc,                \
            int64_t num_bits,                                    \
            bool is_a_8bit);                                     \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *output,                                        \
            const void *input,                                   \
            const void *perm,                                    \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // GPTQ_MARLIN_REPACK_H
