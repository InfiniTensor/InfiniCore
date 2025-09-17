#ifndef __EQUAL_H__
#define __EQUAL_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                         \
    namespace op::equal::NAMESPACE {                                  \
    class Descriptor final : public InfiniopDescriptor {              \
        struct Opaque;                                                \
        Opaque *_opaque;                                              \
        EqualInfo _info;                                              \
        size_t _workspace_size;                                       \
        Descriptor(                                                   \
            infiniDtype_t dtype,                                      \
            EqualInfo info,                                           \
            size_t workspace_size_,                                   \
            Opaque *opaque,                                           \
            infiniDevice_t device_type,                               \
            int device_id                                             \
        ) : InfiniopDescriptor{device_type, device_id},               \
              _opaque(opaque),                                        \
              _info(info),                                            \
              _workspace_size(workspace_size_) {}                     \
    public:                                                           \
        ~Descriptor();                                                \
        size_t workspaceSize() const { return _workspace_size; }      \
        static infiniStatus_t create(                                 \
            infiniopHandle_t handle,                                  \
            Descriptor **desc_ptr,                                    \
            infiniopTensorDescriptor_t c_desc,                        \
            infiniopTensorDescriptor_t a_desc,                        \
            infiniopTensorDescriptor_t b_desc                         \
        );                                                            \
        infiniStatus_t calculate(                                     \
            void *workspace,                                          \
            size_t workspace_size,                                    \
            void * c,                                                 \
            const void * a,                                           \
            const void * b,                                           \
            void *stream                                              \
        ) const;                                                      \
    };                                                                \
    }

#endif