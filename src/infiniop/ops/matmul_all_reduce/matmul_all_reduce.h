#ifndef __MATMUL_ALL_REDUCE_H__
#define __MATMUL_ALL_REDUCE_H__

#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::matmul_all_reduce::NAMESPACE {                 \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            size_t workspace_size,                               \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
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
            infiniopTensorDescriptor_t weight_desc,              \
            infiniopTensorDescriptor_t bias_desc,                \
            const char *group_name);                             \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *input,                                   \
            const void *weight,                                  \
            const void *bias,                                    \
            void *stream) const;                                 \
    };                                                           \
    }

#endif
