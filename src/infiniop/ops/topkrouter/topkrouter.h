#ifndef _Topkrouter_H_
#define _Topkrouter_H_

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::topkrouter::NAMESPACE {                        \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        TopkrouterInfo _info;                                    \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            TopkrouterInfo info,                                 \
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
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t correction_bias_desc,     \
            size_t N, size_t width, size_t topk);                \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            float *values,                                       \
            int *indices,                                        \
            void *x,                                             \
            float *correction_bias,                              \
            float routed_scaling_factor,                         \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // _Topkrouter_H_
