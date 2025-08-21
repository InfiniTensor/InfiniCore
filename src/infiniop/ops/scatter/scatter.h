#ifndef __SCATTER_H__
#define __SCATTER_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                         \
    namespace op::scatter::NAMESPACE {                                \
    class Descriptor final : public InfiniopDescriptor {              \
        struct Opaque;                                                \
        Opaque *_opaque;                                              \
        ScatterInfo _info;                                            \
        size_t _workspace_size;                                       \
        Descriptor(                                                   \
            infiniDtype_t dtype,                                      \
            ScatterInfo info,                                         \
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
            infiniopTensorDescriptor_t output_desc,                   \
            infiniopTensorDescriptor_t input_desc,                    \
            infiniopTensorDescriptor_t index_desc,                    \
            size_t dim                                                \
        );                                                            \
        infiniStatus_t calculate(                                     \
            void *workspace,                                          \
            size_t workspace_size,                                    \
            void * output,                                            \
            const void * input,                                       \
            const void * index,                                       \
            void *stream                                              \
        ) const;                                                      \
    };                                                                \
    }

#endif