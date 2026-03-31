#ifndef __GPTQ_MARLIN_GEMM_H__
#define __GPTQ_MARLIN_GEMM_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::gptq_marlin_gemm::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        GptqMarlinGemmInfo _info;                                \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            GptqMarlinGemmInfo info,                             \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t out_desc,                 \
            infiniopTensorDescriptor_t a_desc,                   \
            infiniopTensorDescriptor_t b_desc,                   \
            infiniopTensorDescriptor_t b_scales_desc,            \
            infiniopTensorDescriptor_t global_scale_desc,        \
            infiniopTensorDescriptor_t b_zeros_desc,             \
            infiniopTensorDescriptor_t g_idx_desc,               \
            infiniopTensorDescriptor_t perm_desc);               \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *out,                                           \
            const void *a,                                       \
            const void *b,                                       \
            void *b_scales,                                      \
            void *global_scale,                                  \
            void *b_zeros,                                       \
            void *g_idx,                                         \
            void *perm,                                          \
            int64_t b_q_type_id,                                 \
            bool is_k_full,                                      \
            bool use_atomic_add,                                 \
            bool use_fp32_reduce,                                \
            bool is_zp_float,                                    \
            void *stream) const;                                 \
    };                                                           \
    }

#endif //__GPTQ_MARLIN_GEMM_H__
