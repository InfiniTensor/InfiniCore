#ifndef SIMPLE_GLA_PREFILL_CUDA_H
#define SIMPLE_GLA_PREFILL_CUDA_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                              \
    namespace op::simple_gla_prefill_cuda::NAMESPACE {                     \
    class Descriptor final : public InfiniopDescriptor {                   \
        struct Opaque;                                                     \
        Opaque *_opaque;                                                   \
        op::simple_gla_prefill_cuda::SimpleGLAPrefillCudaInfo _info;       \
        size_t _workspace_size;                                            \
                                                                           \
        Descriptor(Opaque *opaque,                                         \
                   op::simple_gla_prefill_cuda::SimpleGLAPrefillCudaInfo info, \
                   size_t workspace_size,                                  \
                   infiniDevice_t device_type,                             \
                   int device_id)                                          \
            : InfiniopDescriptor{device_type, device_id},                  \
              _opaque(opaque),                                             \
              _info(info),                                                 \
              _workspace_size(workspace_size) {}                           \
                                                                           \
    public:                                                                \
        ~Descriptor();                                                     \
        size_t workspaceSize() const { return _workspace_size; }           \
        static infiniStatus_t create(                                      \
            infiniopHandle_t handle,                                       \
            Descriptor **desc_ptr,                                         \
            infiniopTensorDescriptor_t out_desc,                           \
            infiniopTensorDescriptor_t q_desc,                             \
            infiniopTensorDescriptor_t k_desc,                             \
            infiniopTensorDescriptor_t v_desc,                             \
            infiniopTensorDescriptor_t g_gamma_desc);                      \
        infiniStatus_t calculate(                                          \
            void *workspace, size_t workspace_size,                        \
            void *out,                                                     \
            const void *q,                                                 \
            const void *k,                                                 \
            const void *v,                                                 \
            const void *g_gamma,                                           \
            float scale,                                                   \
            void *stream) const;                                           \
    };                                                                     \
    }

#endif // SIMPLE_GLA_PREFILL_CUDA_H

