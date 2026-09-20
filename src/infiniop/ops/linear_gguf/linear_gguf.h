#ifndef __LINEAR_GGUF_H__
#define __LINEAR_GGUF_H__

#include "../../operator.h"
#include "info.h"

#include <cstdint>

#define DESCRIPTOR(NAMESPACE)                                     \
    namespace op::linear_gguf::NAMESPACE {                        \
    class Descriptor final : public InfiniopDescriptor {          \
        struct Opaque;                                            \
        Opaque *_opaque;                                          \
        LinearGgufInfo _info;                                     \
        size_t _workspace_size;                                   \
                                                                  \
        Descriptor(                                               \
            Opaque *opaque,                                       \
            LinearGgufInfo info,                                  \
            size_t workspace_size,                                \
            infiniDevice_t device_type,                           \
            int device_id)                                        \
            : InfiniopDescriptor{device_type, device_id},         \
              _opaque(opaque),                                    \
              _info(info),                                        \
              _workspace_size(workspace_size) {}                  \
                                                                  \
    public:                                                       \
        ~Descriptor();                                            \
                                                                  \
        size_t workspaceSize() const { return _workspace_size; }  \
                                                                  \
        static infiniStatus_t create(                             \
            infiniopHandle_t handle,                              \
            Descriptor **desc_ptr,                                \
            infiniopTensorDescriptor_t out_desc,                  \
            infiniopTensorDescriptor_t a_desc,                    \
            infiniopTensorDescriptor_t w_desc,                    \
            int64_t ggml_type);                                   \
                                                                  \
        infiniStatus_t calculate(                                 \
            void *workspace, size_t workspace_size,               \
            void *out,                                            \
            const void *a, const void *w,                         \
            void *stream) const;                                  \
    };                                                            \
    }

#endif // __LINEAR_GGUF_H__
