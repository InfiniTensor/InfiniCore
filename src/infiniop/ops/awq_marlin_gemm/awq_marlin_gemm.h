#ifndef AWQ_MARLIN_GEMM_H
#define AWQ_MARLIN_GEMM_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                 \
                                                                                                              \
    namespace op::awq_marlin_gemm::NAMESPACE {                                                                \
    class Descriptor final : public InfiniopDescriptor {                                                      \
        struct Opaque;                                                                                        \
        Opaque *_opaque;                                                                                      \
        AwqMarlinGemmInfo _info;                                                                              \
        size_t _workspace_size;                                                                               \
                                                                                                              \
        Descriptor(                                                                                           \
            Opaque *opaque,                                                                                   \
            AwqMarlinGemmInfo info,                                                                           \
            size_t workspace_size,                                                                            \
            infiniDevice_t device_type,                                                                       \
            int device_id)                                                                                    \
            : InfiniopDescriptor{device_type, device_id},                                                     \
              _opaque(opaque),                                                                                \
              _info(info),                                                                                    \
              _workspace_size(workspace_size) {}                                                              \
                                                                                                              \
    public:                                                                                                   \
        ~Descriptor();                                                                                        \
                                                                                                              \
        size_t workspaceSize() const { return _workspace_size; }                                              \
                                                                                                              \
        static infiniStatus_t create(                                                                         \
            infiniopHandle_t handle,                                                                          \
            Descriptor **desc_ptr,                                                                            \
            infiniopTensorDescriptor_t out_desc,                                                              \
            infiniopTensorDescriptor_t a_desc,                                                                \
            infiniopTensorDescriptor_t b_desc,                                                                \
            infiniopTensorDescriptor_t b_bias_desc,                                                           \
            infiniopTensorDescriptor_t b_scales_desc,                                                         \
            infiniopTensorDescriptor_t a_scales_desc,                                                         \
            infiniopTensorDescriptor_t global_scales_desc,                                                    \
            infiniopTensorDescriptor_t b_zeros_desc,                                                          \
            infiniopTensorDescriptor_t g_idx_desc,                                                            \
            infiniopTensorDescriptor_t perm_desc);                                                            \
                                                                                                              \
        infiniStatus_t calculate(                                                                             \
            void *workspace, size_t workspace_size,                                                           \
            void *c,                                                                                          \
            const void *a, const void *b,                                                                     \
            void *b_bias, void *b_scales, void *a_scales, void *global_scales,                                \
            void *b_zeros, void *g_idx, void *perm,                                                           \
            int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float, \
            void *stream) const;                                                                              \
    };                                                                                                        \
    }

#endif // AWQ_MARLIN_GEMM_H
