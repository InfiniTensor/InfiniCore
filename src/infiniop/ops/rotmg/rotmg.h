#ifndef __ROTMG_H__
#define __ROTMG_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::rotmg::NAMESPACE {                             \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        RotmgInfo _info;                                         \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            RotmgInfo info,                                      \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(std::move(info)),                            \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t d1_desc,                  \
            infiniopTensorDescriptor_t d2_desc,                  \
            infiniopTensorDescriptor_t x1_desc,                  \
            infiniopTensorDescriptor_t y1_desc,                  \
            infiniopTensorDescriptor_t param_desc);              \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *d1,                                            \
            void *d2,                                            \
            void *x1,                                            \
            const void *y1,                                      \
            void *param,                                         \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __ROTMG_H__
