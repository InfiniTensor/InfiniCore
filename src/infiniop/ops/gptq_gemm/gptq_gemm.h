#ifndef GPTQ_GEMM_H
#define GPTQ_GEMM_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                           \
                                                                                                        \
    namespace op::gptq_gemm::NAMESPACE {                                                                \
    class Descriptor final : public InfiniopDescriptor {                                                \
        struct Opaque;                                                                                  \
        Opaque *_opaque;                                                                                \
        GptqGemmInfo _info;                                                                             \
        size_t _workspace_size;                                                                         \
                                                                                                        \
        Descriptor(                                                                                     \
            Opaque *opaque,                                                                             \
            GptqGemmInfo info,                                                                          \
            size_t workspace_size,                                                                      \
            infiniDevice_t device_type,                                                                 \
            int device_id)                                                                              \
            : InfiniopDescriptor{device_type, device_id},                                               \
              _opaque(opaque),                                                                          \
              _info(info),                                                                              \
              _workspace_size(workspace_size) {}                                                        \
                                                                                                        \
    public:                                                                                             \
        ~Descriptor();                                                                                  \
                                                                                                        \
        size_t workspaceSize() const { return _workspace_size; }                                        \
                                                                                                        \
        static infiniStatus_t create(                                                                   \
            infiniopHandle_t handle,                                                                    \
            Descriptor **desc_ptr,                                                                      \
            infiniopTensorDescriptor_t out_desc,                                                        \
            infiniopTensorDescriptor_t a_desc,                                                          \
            infiniopTensorDescriptor_t b_desc,                                                          \
            infiniopTensorDescriptor_t b_scales_desc,                                                   \
            infiniopTensorDescriptor_t b_zeros_desc,                                                    \
            infiniopTensorDescriptor_t b_g_idx_desc,                                                    \
            bool use_exllama,                                                                           \
            int quant_bit);                                                                             \
                                                                                                        \
        infiniStatus_t calculate(                                                                       \
            void *workspace, size_t workspace_size,                                                     \
            void *out,                                                                                  \
            const void *a, const void *b, const void *b_scale, const void *b_zero, const void *b_g_idx, \
            void *stream) const;                                                                        \
    };                                                                                                  \
    }

#endif // GPTQ_GEMM_H
