#ifndef __LEAKYRELU_H__
#define __LEAKYRELU_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::leakyrelu::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
                                                          \
        LeakyReLUInfo _info;                              \
        size_t _min_workspace_size;                       \
                                                          \
        Descriptor(                                       \
            LeakyReLUInfo info,                           \
            size_t min_workspace_size,                    \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _info(info),                                \
              _min_workspace_size(min_workspace_size) {}  \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t out_desc,          \
            infiniopTensorDescriptor_t in_desc,           \
            float negative_slope);                        \
                                                          \
        size_t workspaceSize() const;                     \
                                                          \
        infiniStatus_t calculate(                         \
            void *workspace,                              \
            size_t workspace_size,                        \
            void *output,                                 \
            const void *input,                            \
            void *stream) const;                          \
    };                                                    \
    }

#endif // __LEAKYRELU_H__
