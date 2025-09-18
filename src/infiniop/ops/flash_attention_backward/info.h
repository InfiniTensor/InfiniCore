#ifndef __FLASH_ATTENTION_BACKWARD_INFO_H__
#define __FLASH_ATTENTION_BACKWARD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/flash_attention.h"
#include <cmath>
#include <vector>

namespace op::flash_attention_backward {

class FlashAttentionBackwardInfo {
private:
    FlashAttentionBackwardInfo() = default;

public:
    infiniDtype_t dtype;
    size_t batch_size;
    size_t seq_len_q, seq_len_kv;
    size_t num_heads_q, num_heads_kv;
    size_t head_dim;

    ptrdiff_t qo_stride_b;
    ptrdiff_t qo_stride_s;
    ptrdiff_t qo_stride_n;
    ptrdiff_t qo_stride_d;

    ptrdiff_t kv_stride_b;
    ptrdiff_t kv_stride_s;
    ptrdiff_t kv_stride_n;
    ptrdiff_t kv_stride_d;

    ptrdiff_t l_stride_b;
    ptrdiff_t l_stride_s;
    ptrdiff_t l_stride_n;

    ptrdiff_t mask_stride_sq;
    ptrdiff_t mask_stride_sk;

    void *mask;
    bool is_masked;

    static utils::Result<FlashAttentionBackwardInfo> create(
        infiniopTensorDescriptor_t grad_q_desc,
        infiniopTensorDescriptor_t grad_k_desc,
        infiniopTensorDescriptor_t grad_v_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t grad_out_desc,
        infiniopTensorDescriptor_t mask_desc,
        infiniopAttentionMaskType_t mask_type) {
        // 检查数据类型是否一致
        auto dtype = grad_out_desc->dtype();
        CHECK_OR_RETURN(
            dtype == grad_q_desc->dtype()
                && dtype == grad_k_desc->dtype()
                && dtype == grad_v_desc->dtype()
                && dtype == q_desc->dtype()
                && dtype == k_desc->dtype()
                && dtype == v_desc->dtype(),
            INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

        // 检查张量形状
        // grad_q, q, grad_out 形状相同
        auto q_shape = q_desc->shape();
        CHECK_SAME_SHAPE(q_shape, grad_q_desc->shape());
        CHECK_SAME_SHAPE(q_shape, grad_out_desc->shape());
        // grad_k, grad_v, k, v 形状相同
        auto kv_shape = k_desc->shape();
        CHECK_SAME_SHAPE(kv_shape, grad_k_desc->shape());
        CHECK_SAME_SHAPE(kv_shape, grad_v_desc->shape());
        CHECK_SAME_SHAPE(kv_shape, v_desc->shape());
        // 检查输入的纬度
        auto ndim = q_desc->ndim();
        CHECK_OR_RETURN(ndim == k_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(ndim == 3 || ndim == 4, INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t batch_size_q = 1;
        size_t seq_len_q = q_shape[ndim - 3];
        size_t num_heads_q = q_shape[ndim - 2];
        size_t head_dim_q = q_shape[ndim - 1];

        size_t batch_size_kv = 1;
        size_t seq_len_kv = kv_shape[ndim - 3];
        size_t num_heads_kv = kv_shape[ndim - 2];
        size_t head_dim_kv = kv_shape[ndim - 1];

        ptrdiff_t qo_stride_b = 0,
                  qo_stride_s = q_desc->stride(ndim - 3),
                  qo_stride_n = q_desc->stride(ndim - 2),
                  qo_stride_d = q_desc->stride(ndim - 1);

        ptrdiff_t kv_stride_b = 0,
                  kv_stride_s = k_desc->stride(ndim - 3),
                  kv_stride_n = k_desc->stride(ndim - 2),
                  kv_stride_d = k_desc->stride(ndim - 1);

        ptrdiff_t l_stride_b = 0,
                  l_stride_s = head_dim_q,
                  l_stride_n = 1;

        if (ndim == 4) {
            qo_stride_b = q_desc->stride(0);
            kv_stride_b = k_desc->stride(0);
            batch_size_q = q_shape[0];
            batch_size_kv = kv_shape[0];

            l_stride_b = seq_len_q * head_dim_q;
        }

        // batch_size 和 head_dim 是否一致
        CHECK_OR_RETURN(batch_size_q == batch_size_kv, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(head_dim_q == head_dim_kv, INFINI_STATUS_BAD_TENSOR_SHAPE);
        // 多头注意力是否整除
        CHECK_OR_RETURN(num_heads_q % num_heads_kv == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(grad_q_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(grad_k_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(grad_v_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(q_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(k_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(v_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(grad_out_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);

        size_t batch_size = batch_size_q;
        size_t head_dim = head_dim_q;

        // 处理不同的 MASK_TYPE
        ptrdiff_t mask_stride_sq = seq_len_kv,
                  mask_stride_sk = 1;
        void *mask = nullptr;
        bool is_masked = true;

        if (mask_type == INFINIOP_ATTENTION_MASK_TYPE_NONE) {
            mask_stride_sq = 0;
            mask_stride_sk = 0;
            is_masked = false;
        } else if (mask_type == INFINIOP_ATTENTION_MASK_TYPE_FULL) {
            auto mask_dtype = mask_desc->dtype();
            CHECK_DTYPE(mask_dtype, INFINI_DTYPE_F32);
            CHECK_OR_RETURN(mask_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
            CHECK_OR_RETURN(mask_desc->dim(0) == seq_len_q && mask_desc->dim(1) == seq_len_kv, INFINI_STATUS_BAD_TENSOR_SHAPE);
            CHECK_OR_RETURN(mask_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        } else if (mask_type == INFINIOP_ATTENTION_MASK_TYPE_CAUSAL) {
            size_t mask_size = seq_len_q * seq_len_kv;
            float *causal_mask = new float[mask_size];

            for (size_t i = 0; i < seq_len_q; ++i) {
                for (size_t j = 0; j < seq_len_kv; ++j) {
                    if (j > i) {
                        causal_mask[i * seq_len_kv + j] = -INFINITY;
                    } else {
                        causal_mask[i * seq_len_kv + j] = 0.0f;
                    }
                }
            }

            mask = causal_mask;
        }

        return utils::Result<FlashAttentionBackwardInfo>(FlashAttentionBackwardInfo{
            dtype,
            batch_size,
            seq_len_q,
            seq_len_kv,
            num_heads_q,
            num_heads_kv,
            head_dim,
            qo_stride_b,
            qo_stride_s,
            qo_stride_n,
            qo_stride_d,
            kv_stride_b,
            kv_stride_s,
            kv_stride_n,
            kv_stride_d,
            l_stride_b,
            l_stride_s,
            l_stride_n,
            mask_stride_sq,
            mask_stride_sk,
            mask,
            is_masked,
        });
    }
};

} // namespace op::flash_attention_backward

#endif // __FLASH_ATTENTION_BACKWARD_INFO_H__
