#ifndef SWIGLU_H
#define SWIGLU_H

#include "../../operator.h"
#include "swiglu_info.h"

#define DESCRIPTOR(NAMESPACE)                                     \
                                                                  \
    namespace op::swiglu::NAMESPACE {                             \
    class Descriptor final : public InfiniopDescriptor {          \
        struct Opaque;                                            \
        Opaque *_opaque;                                          \
        SwiGLUInfo _info;                                        \
        size_t _workspace_size;                                   \
                                                                  \
        Descriptor(                                               \
            Opaque *opaque,                                       \
            SwiGLUInfo info,                                     \
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
            infiniopTensorDescriptor_t c_desc,                    \
            std::vector<infiniopTensorDescriptor_t> input_descs); \
                                                                  \
        infiniStatus_t calculate(                                 \
            void *workspace, size_t workspace_size,               \
            void *c,                                              \
            std::vector<const void *> inputs,                     \
            void *stream) const;                                  \
    };                                                            \
    }

#endif // SWIGLU_H
