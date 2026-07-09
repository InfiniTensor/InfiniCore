#ifndef DSV4_LINEAR_BF16_FP32_H
#define DSV4_LINEAR_BF16_FP32_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                                                                                                  \
    namespace op::dsv4_linear_bf16_fp32::NAMESPACE {                                                                                                                                           \
    class Descriptor final : public InfiniopDescriptor {                                                                                                                                       \
        struct Opaque;                                                                                                                                                                         \
        Info _info;                                                                                                                                                                            \
        size_t _workspace_size;                                                                                                                                                                \
        Opaque *_opaque;                                                                                                                                                                       \
        Descriptor(Info info, size_t workspace_size, Opaque *opaque, infiniDevice_t device_type, int device_id)                                                                                \
            : InfiniopDescriptor{device_type, device_id}, _info(info), _workspace_size(workspace_size), _opaque(opaque) {}                                                                     \
                                                                                                                                                                                               \
    public:                                                                                                                                                                                    \
        ~Descriptor();                                                                                                                                                                         \
        size_t workspaceSize() const { return _workspace_size; }                                                                                                                               \
        const Info &info() const { return _info; }                                                                                                                                             \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t w_desc); \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *y, const void *x, const void *w, void *stream) const;                                                           \
    };                                                                                                                                                                                         \
    }

#endif
