#ifndef __PER_TENSOR_QUANT_FP8_H__
#define __PER_TENSOR_QUANT_FP8_H__

#include "../../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                      \
                                                                                                                   \
    namespace op::per_tensor_quant_fp8::NAMESPACE {                                                               \
    class Descriptor final : public InfiniopDescriptor {                                                           \
        struct Opaque;                                                                                             \
        Opaque *_opaque;                                                                                           \
        PerTensorQuantFp8Info _info;                                                                               \
        size_t _workspace_size;                                                                                    \
                                                                                                                   \
        Descriptor(Opaque *opaque, PerTensorQuantFp8Info info,                                                     \
                   size_t workspace_size,                                                                          \
                   infiniDevice_t device_type, int device_id)                                                      \
            : InfiniopDescriptor{device_type, device_id},                                                          \
              _opaque(opaque), _info(info), _workspace_size(workspace_size) {}                                     \
                                                                                                                   \
    public:                                                                                                        \
        ~Descriptor();                                                                                             \
                                                                                                                   \
        size_t minWorkspaceSize() const { return _workspace_size; }                                                \
                                                                                                                   \
        static infiniStatus_t create(                                                                              \
            infiniopHandle_t handle, Descriptor **desc_ptr,                                                        \
            infiniopTensorDescriptor_t x_packed_desc,                                                              \
            infiniopTensorDescriptor_t x_scale_desc,                                                               \
            infiniopTensorDescriptor_t x_desc);                                                                    \
                                                                                                                   \
        infiniStatus_t calculate(                                                                                  \
            void *workspace, size_t workspace_size,                                                                \
            void *x_packed, void *x_scale, const void *x, const bool is_static, void *stream) const;               \
    };                                                                                                             \
    }

#endif // __PER_TENSOR_QUANT_FP8_H__
