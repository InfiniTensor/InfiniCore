#ifndef __MXFP4_DEQUANTIZE_H__
#define __MXFP4_DEQUANTIZE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                            \
    namespace op::mxfp4_dequantize::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {                 \
        struct Opaque;                                                   \
        Opaque *_opaque;                                                 \
        Mxfp4DequantizeInfo _info;                                       \
                                                                         \
        Descriptor(Opaque *opaque, Mxfp4DequantizeInfo info,             \
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
            infiniopTensorDescriptor_t packed_desc,                      \
            infiniopTensorDescriptor_t scales_desc);                     \
                                                                         \
        infiniStatus_t calculate(                                        \
            void *workspace, size_t workspace_size, void *out,           \
            const void *packed, const void *scales, void *stream) const; \
    };                                                                   \
    }

#endif
