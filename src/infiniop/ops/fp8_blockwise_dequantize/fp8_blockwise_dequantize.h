#ifndef __FP8_BLOCKWISE_DEQUANTIZE_H__
#define __FP8_BLOCKWISE_DEQUANTIZE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                            \
    namespace op::fp8_blockwise_dequantize::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {                 \
        struct Opaque;                                                   \
        Opaque *_opaque;                                                 \
        Fp8BlockwiseDequantizeInfo _info;                                \
                                                                         \
        Descriptor(Opaque *opaque, Fp8BlockwiseDequantizeInfo info,      \
                   infiniDevice_t device_type, int device_id)            \
            : InfiniopDescriptor{device_type, device_id},                \
              _opaque(opaque), _info(info) {}                            \
                                                                         \
    public:                                                              \
        ~Descriptor();                                                   \
        size_t workspaceSize() const { return 0; }                       \
                                                                         \
        static infiniStatus_t create(                                    \
            infiniopHandle_t handle, Descriptor **desc_ptr,              \
            infiniopTensorDescriptor_t out_desc,                         \
            infiniopTensorDescriptor_t q_desc,                           \
            infiniopTensorDescriptor_t scales_desc);                     \
                                                                         \
        infiniStatus_t calculate(                                        \
            void *workspace, size_t workspace_size, void *out,           \
            const void *q, const void *scales, void *stream) const;      \
    };                                                                   \
    }

#endif
