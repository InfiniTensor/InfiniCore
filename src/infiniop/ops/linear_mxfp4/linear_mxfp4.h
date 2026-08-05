#ifndef __LINEAR_MXFP4_H__
#define __LINEAR_MXFP4_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::linear_mxfp4::NAMESPACE {                      \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        LinearMxfp4Info _info;                                   \
                                                                 \
        Descriptor(Opaque *opaque, LinearMxfp4Info info,         \
                   infiniDevice_t device_type, int device_id)    \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque), _info(info) {}                    \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
        size_t workspaceSize() const { return 0; }               \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle, Descriptor **desc_ptr,      \
            infiniopTensorDescriptor_t output_desc,              \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t packed_weight_desc,       \
            infiniopTensorDescriptor_t weight_scale_desc,        \
            infiniopTensorDescriptor_t bias_desc,                \
            float alpha);                                        \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *output, const void *input,                     \
            const void *packed_weight, const void *weight_scale, \
            const void *bias, void *stream) const;               \
    };                                                           \
    }

#endif
