#ifndef __MOE_WNA16_MARLIN_GEMM_H__
#define __MOE_WNA16_MARLIN_GEMM_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                       \
                                                                    \
    namespace op::moe_wna16_marlin_gemm::NAMESPACE {                \
    class Descriptor final : public InfiniopDescriptor {            \
        struct Opaque;                                              \
        Opaque *_opaque;                                            \
        MoeWna16MarlinGemmInfo _info;                               \
        size_t _workspace_size;                                     \
                                                                    \
        Descriptor(                                                 \
            size_t workspace_size_,                                 \
            Opaque *opaque,                                         \
            MoeWna16MarlinGemmInfo info,                            \
            infiniDevice_t device_type,                             \
            int device_id)                                          \
            : InfiniopDescriptor{device_type, device_id},           \
              _opaque(opaque),                                      \
              _info(info),                                          \
              _workspace_size(workspace_size_) {}                   \
                                                                    \
    public:                                                         \
        ~Descriptor();                                              \
                                                                    \
        size_t workspaceSize() const { return _workspace_size; }    \
                                                                    \
        static infiniStatus_t create(                               \
            infiniopHandle_t handle,                                \
            Descriptor **desc_ptr,                                  \
            infiniopTensorDescriptor_t c_desc,                      \
            infiniopTensorDescriptor_t a_desc,                      \
            infiniopTensorDescriptor_t b_q_weight_desc,             \
            infiniopTensorDescriptor_t b_bias_desc,                 \
            infiniopTensorDescriptor_t b_scales_desc,               \
            infiniopTensorDescriptor_t global_scales_desc,          \
            infiniopTensorDescriptor_t b_zeros_desc,                \
            infiniopTensorDescriptor_t g_idx_desc,                  \
            infiniopTensorDescriptor_t perm_desc,                   \
            infiniopTensorDescriptor_t sorted_token_desc,           \
            infiniopTensorDescriptor_t expert_ids_desc,             \
            infiniopTensorDescriptor_t num_tokens_post_padded_desc, \
            infiniopTensorDescriptor_t topk_weights_desc,           \
            int size_m,                                             \
            int size_n,                                             \
            int size_k,                                             \
            int top_k,                                              \
            int moe_block_size);                                    \
                                                                    \
        infiniStatus_t calculate(                                   \
            void *workspace,                                        \
            size_t workspace_size,                                  \
            void *c,                                                \
            const void *a,                                          \
            const void *b_q_weight,                                 \
            void *b_bias,                                           \
            void *b_scales,                                         \
            void *global_scales,                                    \
            void *b_zeros,                                          \
            void *g_idx,                                            \
            void *perm,                                             \
            void *sorted_token_ids,                                 \
            void *expert_ids,                                       \
            void *num_tokens_post_padded,                           \
            void *topk_weights,                                     \
            bool mul_topk_weights,                                  \
            bool is_ep,                                             \
            int64_t b_q_type_id,                                    \
            bool is_k_full,                                         \
            bool use_atomic_add,                                    \
            bool use_fp32_reduce,                                   \
            bool is_zp_float,                                       \
            void *stream) const;                                    \
    };                                                              \
    }

#endif //__MOE_WNA16_MARLIN_GEMM_H__
