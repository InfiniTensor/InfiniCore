#ifndef __CAST_H__
#define __CAST_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::cast::NAMESPACE {                       \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
                                                          \
        CastInfo _info;                                   \
        size_t _min_workspace_size;                       \
                                                          \
        Descriptor(                                       \
            CastInfo info,                                \
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
            infiniopTensorDescriptor_t in_desc);          \
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

#endif // __CAST_H__
