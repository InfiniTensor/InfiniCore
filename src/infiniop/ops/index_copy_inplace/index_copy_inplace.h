#ifndef __INDEX_COPY_INPLACE_H__
#define __INDEX_COPY_INPLACE_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "info.h"

#define INDEX_COPY_INPLACE_DESCRIPTOR(NAMESPACE)                                         \
    namespace op::index_copy_inplace::NAMESPACE {                     \
    class Descriptor final : public InfiniopDescriptor {              \
        struct Opaque;                                                \
        Opaque *_opaque;                                              \
        IndexCopyInplaceInfo _info;                                   \
        size_t _workspace_size;                                       \
        void *_rearrange_desc_in;                                     \
        void *_rearrange_desc_out;                                    \
        Descriptor(                                                   \
            infiniDtype_t dtype,                                      \
            IndexCopyInplaceInfo info,                                \
            size_t workspace_size_,                                   \
            Opaque *opaque,                                           \
            infiniDevice_t device_type,                               \
            int device_id,                                            \
            void *rearrange_desc_in,                                  \
            void *rearrange_desc_out                                  \
        ) : InfiniopDescriptor{device_type, device_id},               \
              _opaque(opaque),                                        \
              _info(info),                                            \
              _workspace_size(workspace_size_),                       \
              _rearrange_desc_in(rearrange_desc_in),                  \
              _rearrange_desc_out(rearrange_desc_out) {}              \
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