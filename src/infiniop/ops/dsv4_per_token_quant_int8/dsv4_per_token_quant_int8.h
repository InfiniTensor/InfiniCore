#ifndef DSV4_PER_TOKEN_QUANT_INT8_H
#define DSV4_PER_TOKEN_QUANT_INT8_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                                                                                                      \
    namespace op::dsv4_per_token_quant_int8::NAMESPACE {                                                                                                                                           \
    class Descriptor final : public InfiniopDescriptor {                                                                                                                                           \
        Info _info;                                                                                                                                                                                \
        size_t _workspace_size;                                                                                                                                                                    \
        Descriptor(Info info, size_t workspace_size, infiniDevice_t device_type, int device_id) : InfiniopDescriptor{device_type, device_id}, _info(info), _workspace_size(workspace_size) {}      \
                                                                                                                                                                                                   \
    public:                                                                                                                                                                                        \
        size_t workspaceSize() const { return _workspace_size; }                                                                                                                                   \
        const Info &info() const { return _info; }                                                                                                                                                 \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t x_desc); \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *q, void *scale, const void *x, void *stream) const;                                                                 \
    };                                                                                                                                                                                             \
    }

#endif
