#ifndef __BLOCK_FP8_LINEAR_H__
#define __BLOCK_FP8_LINEAR_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::block_fp8_linear::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        BlockFP8LinearInfo _info;                                \
                                                                 \
        Descriptor(Opaque *opaque, BlockFP8LinearInfo info,      \
                   infiniDevice_t device_type, int device_id)    \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque), _info(info) {}                    \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
        size_t workspaceSize() const;                            \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle, Descriptor **desc_ptr,      \
            infiniopTensorDescriptor_t output_desc,              \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t weight_desc,              \
            infiniopTensorDescriptor_t weight_scale_desc);        \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *output, const void *input,                     \
            const void *weight, const void *weight_scale,        \
            void *stream) const;                                 \
    };                                                           \
    }

#endif
