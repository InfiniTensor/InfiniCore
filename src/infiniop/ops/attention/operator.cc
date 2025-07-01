#if defined(ENABLE_NINETOOTHED) && (defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API))
#include "../../../../build/ninetoothed/scaled_dot_product_attention.h"
#endif

#include "../../../utils.h"
#include "../../../utils/check.h"
#include "../../handle.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/attention.h"
#include "infiniop/ops/causal_softmax.h"
#include "infiniop/ops/gemm.h"
#include "infiniop/ops/rearrange.h"

#include <cmath>
#include <cstdint>

#if !defined(ENABLE_NINETOOTHED) || (!defined(ENABLE_NVIDIA_API) && !defined(ENABLE_ILUVATAR_API))
struct InfiniopAttentionDescriptor {
    InfiniopDescriptor _super;
    infiniopRearrangeDescriptor_t rearrange_desc_k;
    infiniopRearrangeDescriptor_t rearrange_desc_v;
    infiniopRearrangeDescriptor_t rearrange_desc_q;
    infiniopRearrangeDescriptor_t rearrange_desc_out;
    infiniopGemmDescriptor_t matmul_desc1;
    infiniopGemmDescriptor_t matmul_desc2;
    infiniopCausalSoftmaxDescriptor_t softmax_desc;
    size_t workspace_size;
    size_t op_workspace_offset;
    size_t op_workspace_size;
    size_t q_cont_offset;
    size_t att_score_offset;
    size_t att_val_offset;
    size_t k_cache_offset;
    size_t v_cache_offset;
    float qk_alpha;
};
#else
struct InfiniopAttentionDescriptor {
    InfiniopDescriptor _super;
    size_t workspace_size;
    uint64_t query_shape[4];
    int64_t query_strides[4];
    uint64_t key_shape[4];
    int64_t key_strides[4];
    uint64_t value_shape[4];
    int64_t value_strides[4];
    uint64_t present_key_shape[4];
    int64_t present_key_strides[4];
    uint64_t present_value_shape[4];
    int64_t present_value_strides[4];
    uint64_t present_key_slot_shape[4];
    int64_t present_key_slot_strides[4];
    uint64_t present_value_slot_shape[4];
    int64_t present_value_slot_strides[4];
    double scale;
    uint64_t output_shape[4];
    int64_t output_strides[4];
    size_t present_key_slot_offset;
    size_t present_value_slot_offset;
    infiniDtype_t dtype;
};
#endif

__C __export infiniStatus_t infiniopCreateAttentionDescriptor(infiniopHandle_t handle,
                                                              infiniopAttentionDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t out_desc,
                                                              infiniopTensorDescriptor_t q_desc,
                                                              infiniopTensorDescriptor_t k_desc,
                                                              infiniopTensorDescriptor_t v_desc,
                                                              infiniopTensorDescriptor_t k_cache_desc,
                                                              infiniopTensorDescriptor_t v_cache_desc,
                                                              size_t pos) {
    if (out_desc->ndim() != 3 || q_desc->ndim() != 3 || k_desc->ndim() != 3 || v_desc->ndim() != 3 || k_cache_desc->ndim() != 3 || v_cache_desc->ndim() != 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (!out_desc->isContiguous()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    if (q_desc->strides()[2] != 1 || k_desc->strides()[2] != 1 || v_desc->strides()[2] != 1 || k_cache_desc->strides()[2] != 1 || v_cache_desc->strides()[2] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    size_t n_q_head = q_desc->shape()[0];
    size_t seq_len = q_desc->shape()[1];
    size_t head_dim = q_desc->shape()[2];
    size_t hidden_size = n_q_head * head_dim;
    size_t n_kv_head = k_desc->shape()[0];
    size_t total_seq_len = seq_len + pos;
    size_t n_group = n_q_head / n_kv_head;
    size_t alignment = 256;

    if (out_desc->shape()[0] != seq_len || out_desc->shape()[1] != n_q_head || out_desc->shape()[2] != head_dim) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // k: [n_kv_head, seq_len, head_dim]
    if (k_desc->shape()[0] != n_kv_head || k_desc->shape()[1] != seq_len || k_desc->shape()[2] != head_dim) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // v: [n_kv_head, seq_len, head_dim]
    if (v_desc->shape()[0] != n_kv_head || v_desc->shape()[1] != seq_len || v_desc->shape()[2] != head_dim) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // k_cache: [n_kv_head, _, head_dim]
    if (k_cache_desc->shape()[0] != n_kv_head || k_cache_desc->shape()[1] < total_seq_len || k_cache_desc->shape()[2] != head_dim) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // v_cache: [n_kv_head, _, head_dim]
    if (v_cache_desc->shape()[0] != n_kv_head || v_cache_desc->shape()[1] < total_seq_len || v_cache_desc->shape()[2] != head_dim) {
        return INFINI_STATUS_BAD_PARAM;
    }

#if !defined(ENABLE_NINETOOTHED) || (!defined(ENABLE_NVIDIA_API) && !defined(ENABLE_ILUVATAR_API))
    // Rearrange k into k_cache
    infiniopTensorDescriptor_t dst_k_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&dst_k_desc, 3, k_desc->shape().data(), k_cache_desc->strides().data(), k_cache_desc->dtype()));
    infiniopRearrangeDescriptor_t rearrange_desc_k;
    CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &rearrange_desc_k, dst_k_desc, k_desc));

    // Rearrange v into v_cache
    infiniopTensorDescriptor_t dst_v_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&dst_v_desc, 3, v_desc->shape().data(), v_cache_desc->strides().data(), v_cache_desc->dtype()));
    infiniopRearrangeDescriptor_t rearrange_desc_v;
    CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &rearrange_desc_v, dst_v_desc, v_desc));

    infiniopRearrangeDescriptor_t rearrange_desc_q = nullptr;
    size_t q_cont_size = 0;
    infiniopTensorDescriptor_t rearranged_q_desc;
    // Rearrange q into contiguous
    if (!q_desc->isContiguous(0, 1)) {
        CHECK_STATUS(infiniopCreateTensorDescriptor(&rearranged_q_desc, 3, q_desc->shape().data(), nullptr, q_desc->dtype()));
        q_cont_size = utils::align(rearranged_q_desc->numel() * infiniSizeOf(rearranged_q_desc->dtype()), alignment);
        rearrange_desc_q = new InfiniopDescriptor;
        CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &rearrange_desc_q, rearranged_q_desc, q_desc));
    }

    // Matmul1: q * full_k
    //      q: [n_q_head, seq_len, head_dim] -> [n_kv_head, n_group *seq_len, head_dim]
    infiniopTensorDescriptor_t reshaped_q_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&reshaped_q_desc, 3, q_desc->shape().data(), nullptr, q_desc->dtype()));
    TRANSFORM_TENSOR_DESC(reshaped_q_desc, dimSplit(0, {n_kv_head, n_group}));
    TRANSFORM_TENSOR_DESC(reshaped_q_desc, dimMerge(1, 2));
    //      full_k: [n_kv_head, head_dim, total_seq_len]
    infiniopTensorDescriptor_t full_k_desc;
    size_t full_k_shape[3] = {n_kv_head, total_seq_len, head_dim};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&full_k_desc, 3, full_k_shape, k_cache_desc->strides().data(), k_cache_desc->dtype()));
    TRANSFORM_TENSOR_DESC(full_k_desc, dimPermute({0, 2, 1}));
    //      qk: [n_kv_head, n_group * seq_len, total_seq_len]
    infiniopTensorDescriptor_t qk_desc;
    size_t qk_shape[3] = {n_kv_head, n_group * seq_len, total_seq_len};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&qk_desc, 3, qk_shape, nullptr, q_desc->dtype()));
    //      matmul1_desc
    //          qk_alpha
    float qk_alpha = 1 / sqrt(head_dim);
    infiniopGemmDescriptor_t matmul1_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &matmul1_desc, qk_desc, reshaped_q_desc, full_k_desc));
    //      matmul1 workspace size
    size_t matmul1_workspace_size;
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(matmul1_desc, &matmul1_workspace_size));
    matmul1_workspace_size = utils::align(matmul1_workspace_size, alignment);
    //      attention score tensor size
    size_t attn_score_size = utils::align(qk_desc->numel() * infiniSizeOf(qk_desc->dtype()), alignment);

    // CausalSoftmax: softmax(qk)
    //      qk: [n_kv_head, n_group * seq_len, total_seq_len] -> [n_q_head, seq_len, total_seq_len]
    TRANSFORM_TENSOR_DESC(qk_desc, dimSplit(1, {n_group, seq_len}));
    TRANSFORM_TENSOR_DESC(qk_desc, dimMerge(0, 1));
    infiniopCausalSoftmaxDescriptor_t softmax_desc;
    CHECK_STATUS(infiniopCreateCausalSoftmaxDescriptor(handle, &softmax_desc, qk_desc, qk_desc));
    //      softmax workspace size
    size_t softmax_workspace_size;
    CHECK_STATUS(infiniopGetCausalSoftmaxWorkspaceSize(softmax_desc, &softmax_workspace_size));
    softmax_workspace_size = utils::align(softmax_workspace_size, alignment);

    // Matmul2: softmax(qk) * full_v
    //      softmax(qk): [n_q_head, seq_len, total_seq_len] -> [n_kv_head, n_group * seq_len, total_seq_len]
    //      full_v: [n_kv_head, total_seq_len, head_dim]
    TRANSFORM_TENSOR_DESC(qk_desc, dimSplit(0, {n_kv_head, n_group}));
    TRANSFORM_TENSOR_DESC(qk_desc, dimMerge(1, 2));
    infiniopTensorDescriptor_t full_v_desc;
    size_t full_v_shape[3] = {n_kv_head, total_seq_len, head_dim};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&full_v_desc, 3, full_v_shape, v_cache_desc->strides().data(), v_cache_desc->dtype()));
    //      temp_out: [n_kv_head, n_group * seq_len, head_dim]
    infiniopTensorDescriptor_t att_val_desc;
    size_t temp_out_shape[3] = {n_kv_head, n_group * seq_len, head_dim};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&att_val_desc, 3, temp_out_shape, nullptr, q_desc->dtype()));
    //      matmul2_desc
    infiniopGemmDescriptor_t matmul2_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &matmul2_desc, att_val_desc, qk_desc, full_v_desc));
    //      matmul2 workspace size
    size_t matmul2_workspace_size;
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(matmul2_desc, &matmul2_workspace_size));
    matmul2_workspace_size = utils::align(matmul2_workspace_size, alignment);
    //      attention value tensor size
    size_t att_val_size = utils::align(att_val_desc->numel() * infiniSizeOf(att_val_desc->dtype()), alignment);

    // Rearrange temp_out into out
    //      out: [seq_len, n_q_head, head_dim]
    //      temp_out: [n_kv_head, n_group * seq_len, head_dim] -> [n_q_head, seq_len, head_dim] -> [seq_len, n_q_head, head_dim]
    TRANSFORM_TENSOR_DESC(att_val_desc, dimSplit(1, {n_group, seq_len}));
    TRANSFORM_TENSOR_DESC(att_val_desc, dimMerge(0, 1));
    TRANSFORM_TENSOR_DESC(att_val_desc, dimPermute({1, 0, 2}));
    infiniopRearrangeDescriptor_t rearrange_desc_out;
    CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &rearrange_desc_out, out_desc, att_val_desc));

    // workspace size
    size_t op_workspace_size = utils::align(std::max(std::max(matmul1_workspace_size, matmul2_workspace_size), softmax_workspace_size), alignment);
    size_t temp_tensors_size = attn_score_size + std::max(q_cont_size, att_val_size);
    size_t workspace_size = temp_tensors_size + op_workspace_size;
#endif

    // k_cache_offset
    size_t k_cache_offset = 0;
    if (pos > 0) {
        k_cache_offset = pos * k_cache_desc->getByteStrides()[1];
    }

    // v_cache_offset
    size_t v_cache_offset = 0;
    if (pos > 0) {
        v_cache_offset = pos * v_cache_desc->getByteStrides()[1];
    }

#if !defined(ENABLE_NINETOOTHED) || (!defined(ENABLE_NVIDIA_API) && !defined(ENABLE_ILUVATAR_API))
    // create attention descriptor
    *(InfiniopAttentionDescriptor **)desc_ptr = new InfiniopAttentionDescriptor{
        {handle->device, handle->device_id},
        rearrange_desc_k,
        rearrange_desc_v,
        rearrange_desc_q,
        rearrange_desc_out,
        matmul1_desc,
        matmul2_desc,
        softmax_desc,
        workspace_size,
        temp_tensors_size,
        op_workspace_size,
        attn_score_size,
        0,
        attn_score_size,
        k_cache_offset,
        v_cache_offset,
        1.f / std::sqrt(float(head_dim)),
    };
#else
    *(InfiniopAttentionDescriptor **)desc_ptr = new InfiniopAttentionDescriptor{
        {handle->device, handle->device_id},
        0,
        {1, q_desc->shape()[0], q_desc->shape()[1], q_desc->shape()[2]},
        {0, q_desc->strides()[0], q_desc->strides()[1], q_desc->strides()[2]},
        {1, k_cache_desc->shape()[0], pos + k_desc->shape()[1], k_cache_desc->shape()[2]},
        {0, k_cache_desc->strides()[0], k_cache_desc->strides()[1], k_cache_desc->strides()[2]},
        {1, v_cache_desc->shape()[0], pos + v_desc->shape()[1], v_cache_desc->shape()[2]},
        {0, v_cache_desc->strides()[0], v_cache_desc->strides()[1], v_cache_desc->strides()[2]},
        {1, k_desc->shape()[0], k_desc->shape()[1], k_desc->shape()[2]},
        {0, k_desc->strides()[0], k_desc->strides()[1], k_desc->strides()[2]},
        {1, v_desc->shape()[0], v_desc->shape()[1], v_desc->shape()[2]},
        {0, v_desc->strides()[0], v_desc->strides()[1], v_desc->strides()[2]},
        {1, k_desc->shape()[0], k_desc->shape()[1], k_desc->shape()[2]},
        {0, k_cache_desc->strides()[0], k_cache_desc->strides()[1], k_cache_desc->strides()[2]},
        {1, v_desc->shape()[0], v_desc->shape()[1], v_desc->shape()[2]},
        {0, v_cache_desc->strides()[0], v_cache_desc->strides()[1], v_cache_desc->strides()[2]},
        static_cast<double>(1.0 / std::sqrt(head_dim)),
        {1, out_desc->shape()[1], out_desc->shape()[0], out_desc->shape()[2]},
        {0, out_desc->strides()[1], out_desc->strides()[0], out_desc->strides()[2]},
        k_cache_offset,
        v_cache_offset,
        out_desc->dtype()};
#endif

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopGetAttentionWorkspaceSize(infiniopAttentionDescriptor_t desc, size_t *size) {
    *size = ((InfiniopAttentionDescriptor *)desc)->workspace_size;
    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopAttention(infiniopAttentionDescriptor_t desc_,
                                              void *workspace_,
                                              size_t workspace_size_,
                                              void *out,
                                              void const *q,
                                              void const *k,
                                              void const *v,
                                              void *k_cache,
                                              void *v_cache,
                                              void *stream) {
    auto desc = (InfiniopAttentionDescriptor *)desc_;
#if !defined(ENABLE_NINETOOTHED) || (!defined(ENABLE_NVIDIA_API) && !defined(ENABLE_ILUVATAR_API))
    if (workspace_size_ < desc->workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE; // STATUS_MEMORY_NOT_ALLOCATED
    }
    void *workspace = (char *)workspace_ + desc->op_workspace_offset;
    size_t workspace_size = desc->op_workspace_size;
    void *att_score = (char *)workspace_ + desc->att_score_offset;
    void *att_val = (char *)workspace_ + desc->att_val_offset;
    void const *q_ = q;
    // concat k and v to k_cache and v_cache
    CHECK_STATUS(infiniopRearrange(desc->rearrange_desc_k,
                                   (char *)k_cache + desc->k_cache_offset, k, stream));

    CHECK_STATUS(infiniopRearrange(desc->rearrange_desc_v,
                                   (char *)v_cache + desc->v_cache_offset, v, stream));

    // rearrange q into contiguous
    if (desc->rearrange_desc_q) {
        void *q_cont = (char *)workspace_ + desc->q_cont_offset;
        CHECK_STATUS(infiniopRearrange(desc->rearrange_desc_q, q_cont, q, stream));
        q_ = q_cont;
    }

    // matmul1: q * full_k
    CHECK_STATUS(infiniopGemm(desc->matmul_desc1,
                              workspace, workspace_size,
                              att_score, q_, k_cache, desc->qk_alpha, 0.0, stream));
    // softmax(qk)
    CHECK_STATUS(infiniopCausalSoftmax(desc->softmax_desc,
                                       workspace, workspace_size,
                                       att_score, att_score, stream));
    // matmul2: softmax(qk) * full_v
    CHECK_STATUS(infiniopGemm(desc->matmul_desc2,
                              workspace, workspace_size,
                              att_val, att_score, v_cache, 1.0, 0.0, stream));
    // rearrange out
    CHECK_STATUS(infiniopRearrange(desc->rearrange_desc_out, out, att_val, stream));
#else
    const auto &query_shape{desc->query_shape};
    const auto &query_strides{desc->query_strides};
    const auto &key_shape{desc->key_shape};
    const auto &key_strides{desc->key_strides};
    const auto &value_shape{desc->value_shape};
    const auto &value_strides{desc->value_strides};
    const auto &present_key_shape{desc->present_key_shape};
    const auto &present_key_strides{desc->present_key_strides};
    const auto &present_value_shape{desc->present_value_shape};
    const auto &present_value_strides{desc->present_value_strides};
    const auto &present_key_slot_shape{desc->present_key_slot_shape};
    const auto &present_key_slot_strides{desc->present_key_slot_strides};
    const auto &present_value_slot_shape{desc->present_value_slot_shape};
    const auto &present_value_slot_strides{desc->present_value_slot_strides};
    const auto &scale_{desc->scale};
    const auto &output_shape{desc->output_shape};
    const auto &output_strides{desc->output_strides};
    const auto &present_key_slot_offset{desc->present_key_slot_offset};
    const auto &present_value_slot_offset{desc->present_value_slot_offset};
    const auto &dtype{desc->dtype};

    uint64_t empty_shape[4];
    int64_t empty_strides[4];

    NineToothedTensor query{const_cast<void *>(q), const_cast<uint64_t *>(query_shape), const_cast<int64_t *>(query_strides)};
    NineToothedTensor key{const_cast<void *>(k_cache), const_cast<uint64_t *>(key_shape), const_cast<int64_t *>(key_strides)};
    NineToothedTensor value{const_cast<void *>(v_cache), const_cast<uint64_t *>(value_shape), const_cast<int64_t *>(value_strides)};
    NineToothedTensor present_key{const_cast<void *>(k), const_cast<uint64_t *>(present_key_shape), const_cast<int64_t *>(present_key_strides)};
    NineToothedTensor present_value{const_cast<void *>(v), const_cast<uint64_t *>(present_value_shape), const_cast<int64_t *>(present_value_strides)};
    NineToothedTensor present_key_slot{const_cast<void *>(static_cast<void *>(static_cast<char *>(k_cache) + present_key_slot_offset)), const_cast<uint64_t *>(present_key_slot_shape), const_cast<int64_t *>(present_key_slot_strides)};
    NineToothedTensor present_value_slot{const_cast<void *>(static_cast<void *>(static_cast<char *>(v_cache) + present_value_slot_offset)), const_cast<uint64_t *>(present_value_slot_shape), const_cast<int64_t *>(present_value_slot_strides)};
    NineToothedTensor attn_mask{nullptr, empty_shape, empty_strides};
    NineToothedTensor is_causal{nullptr, nullptr, nullptr};
    NineToothedTensor scale{const_cast<double *>(&scale_), empty_shape, empty_strides};
    NineToothedTensor output{const_cast<void *>(out), const_cast<uint64_t *>(output_shape), const_cast<int64_t *>(output_strides)};
    NineToothedTensor with_attn_mask{nullptr, nullptr, nullptr};
    NineToothedTensor causal_variant{nullptr, nullptr, nullptr};

    if (launch_scaled_dot_product_attention(stream,
                                            query,
                                            key,
                                            value,
                                            present_key,
                                            present_value,
                                            present_key_slot,
                                            present_value_slot,
                                            attn_mask,
                                            is_causal,
                                            scale,
                                            output,
                                            with_attn_mask,
                                            causal_variant,
                                            1,
                                            output.shape[3],
                                            1,
                                            0,
                                            2,
                                            dtype,
                                            32,
                                            32)) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
#endif

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopDestroyAttentionDescriptor(infiniopAttentionDescriptor_t desc_) {
    auto desc = (InfiniopAttentionDescriptor *)desc_;
#if !defined(ENABLE_NINETOOTHED) || (!defined(ENABLE_NVIDIA_API) && !defined(ENABLE_ILUVATAR_API))
    if (desc->rearrange_desc_q) {
        CHECK_STATUS(infiniopDestroyRearrangeDescriptor(desc->rearrange_desc_q));
    }
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(desc->rearrange_desc_k));
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(desc->rearrange_desc_v));
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(desc->rearrange_desc_out));
    CHECK_STATUS(infiniopDestroyGemmDescriptor(desc->matmul_desc1));
    CHECK_STATUS(infiniopDestroyGemmDescriptor(desc->matmul_desc2));
    CHECK_STATUS(infiniopDestroyCausalSoftmaxDescriptor(desc->softmax_desc));
#endif
    delete desc;

    return INFINI_STATUS_SUCCESS;
}
