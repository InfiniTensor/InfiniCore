#include "../../operator.h"
#include "../../../utils.h"
#include "../../../utils/check.h"
#include "../../handle.h"
#include "../../tensor.h"
#include "infiniop/ops/attention.h"
#include "infiniop/ops/causal_softmax.h"
#include "infiniop/ops/gemm.h"
#include "infiniop/ops/rearrange.h"

#include <cmath>
#include <cstdint>

struct _infiniopAttentionDescriptor {
    infiniDevice_t device;
    infiniopRearrangeDescriptor_t rearrange_desc_k;
    infiniopRearrangeDescriptor_t rearrange_desc_v;
    infiniopRearrangeDescriptor_t rearrange_desc_q;
    infiniopRearrangeDescriptor_t rearrange_desc_out;
    infiniopGemmDescriptor_t matmul_desc1;
    infiniopGemmDescriptor_t matmul_desc2;
    infiniopCausalSoftmaxDescriptor_t softmax_desc;
    uint64_t workspace_size;
    uint64_t rearranged_q_size;
    uint64_t matmul1_workspace_size;
    uint64_t matmul1_tensor_size;
    uint64_t matmul2_workspace_size;
    uint64_t matmul2_tensor_size;
    uint64_t softmax_workspace_size;
    uint64_t k_cache_offset;
    uint64_t v_cache_offset;
};

typedef struct _infiniopAttentionDescriptor *_infiniopAttentionDescriptor_t;

__C __export infiniStatus_t infiniopCreateAttentionDescriptor(infiniopHandle_t handle,
                                                              infiniopAttentionDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t out_desc,
                                                              infiniopTensorDescriptor_t q_desc,
                                                              infiniopTensorDescriptor_t k_desc,
                                                              infiniopTensorDescriptor_t v_desc,
                                                              infiniopTensorDescriptor_t k_cache_desc,
                                                              infiniopTensorDescriptor_t v_cache_desc,
                                                              uint64_t pos) {
    if (out_desc->ndim() != 3 || q_desc->ndim() != 3 || k_desc->ndim() != 3 || v_desc->ndim() != 3 || k_cache_desc->ndim() != 3 || v_cache_desc->ndim() != 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (!out_desc->isContiguous(0, 2)) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    if (q_desc->strides()[2] != 1 || k_desc->strides()[2] != 1 || v_desc->strides()[2] != 1 || k_cache_desc->strides()[2] != 1 || v_cache_desc->strides()[2] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    uint64_t n_q_head = q_desc->shape()[0];
    uint64_t seq_len = q_desc->shape()[1];
    uint64_t head_dim = q_desc->shape()[2];
    uint64_t hidden_size = n_q_head * head_dim;
    uint64_t n_kv_head = k_desc->shape()[0];
    uint64_t total_seq_len = seq_len + pos;
    uint64_t n_group = n_q_head / n_kv_head;

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
    uint64_t rearranged_q_size = 0;
    infiniopTensorDescriptor_t rearranged_q_desc;
    // Rearrange q into contiguous
    if (!q_desc->isContiguous(0, 1)) {
        CHECK_STATUS(infiniopCreateTensorDescriptor(&rearranged_q_desc, 3, q_desc->shape().data(), nullptr, q_desc->dtype()));
        rearranged_q_size = rearranged_q_desc->numel() * infiniSizeOf(rearranged_q_desc->dtype());
        rearrange_desc_q = new InfiniopDescriptor;
        CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &rearrange_desc_q, rearranged_q_desc, q_desc));
    }

    // Matmul1: q * full_k
    //      q: [n_q_head, seq_len, head_dim] -> [n_kv_head, n_group *seq_len, head_dim]
    infiniopTensorDescriptor_t reshaped_q_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&reshaped_q_desc, 3, q_desc->shape().data(), nullptr, q_desc->dtype()));
    reshaped_q_desc = reshaped_q_desc->dimSplit(0, {n_kv_head, n_group});
    if (!reshaped_q_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    reshaped_q_desc = reshaped_q_desc->dimMerge(1, 2);
    if (!reshaped_q_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    //      full_k: [n_kv_head, head_dim, total_seq_len]
    infiniopTensorDescriptor_t full_k_desc;
    uint64_t full_k_shape[3] = {n_kv_head, total_seq_len, head_dim};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&full_k_desc, 3, full_k_shape, k_cache_desc->strides().data(), k_cache_desc->dtype()));
    full_k_desc = full_k_desc->dimPermute({0, 2, 1});
    if (!full_k_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    //      qk: [n_kv_head, n_group * seq_len, total_seq_len]
    infiniopTensorDescriptor_t qk_desc;
    uint64_t qk_shape[3] = {n_kv_head, n_group * seq_len, total_seq_len};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&qk_desc, 3, qk_shape, nullptr, q_desc->dtype()));
    //      matmul1_desc
    //          qk_alpha
    float qk_alpha = 1 / sqrt(head_dim);
    infiniopGemmDescriptor_t matmul1_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &matmul1_desc, qk_desc, reshaped_q_desc, full_k_desc));
    //      matmul1 workspace size
    uint64_t matmul1_workspace_size;
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(matmul1_desc, &matmul1_workspace_size));
    //      matmul1 tensor size
    uint64_t matmul1_tensor_size = qk_desc->numel() * infiniSizeOf(qk_desc->dtype());

    // CausalSoftmax: softmax(qk)
    //      qk: [n_kv_head, n_group * seq_len, total_seq_len] -> [n_q_head, seq_len, total_seq_len]
    qk_desc = qk_desc->dimSplit(1, {n_group, seq_len});
    if (!qk_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    qk_desc = qk_desc->dimMerge(0, 1);
    if (!qk_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    infiniopCausalSoftmaxDescriptor_t softmax_desc;
    CHECK_STATUS(infiniopCreateCausalSoftmaxDescriptor(handle, &softmax_desc, qk_desc, qk_desc));
    //      softmax workspace size
    uint64_t softmax_workspace_size;
    CHECK_STATUS(infiniopGetCausalSoftmaxWorkspaceSize(softmax_desc, &softmax_workspace_size));

    // Matmul2: softmax(qk) * full_v
    //      softmax(qk): [n_q_head, seq_len, total_seq_len] -> [n_kv_head, n_group * seq_len, total_seq_len]
    //      full_v: [n_kv_head, total_seq_len, head_dim]
    qk_desc = qk_desc->dimSplit(0, {n_kv_head, n_group});
    if (!qk_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    qk_desc = qk_desc->dimMerge(1, 2);
    if (!qk_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    infiniopTensorDescriptor_t full_v_desc;
    uint64_t full_v_shape[3] = {n_kv_head, total_seq_len, head_dim};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&full_v_desc, 3, full_v_shape, v_cache_desc->strides().data(), v_cache_desc->dtype()));
    //      temp_out: [n_kv_head, n_group * seq_len, head_dim]
    infiniopTensorDescriptor_t temp_out_desc;
    uint64_t temp_out_shape[3] = {n_kv_head, n_group * seq_len, head_dim};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&temp_out_desc, 3, temp_out_shape, nullptr, q_desc->dtype()));
    //      matmul2_desc
    infiniopGemmDescriptor_t matmul2_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &matmul2_desc, temp_out_desc, qk_desc, full_v_desc));
    //      matmul2 workspace size
    uint64_t matmul2_workspace_size;
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(matmul2_desc, &matmul2_workspace_size));
    //      matmul2 tensor size
    uint64_t matmul2_tensor_size = temp_out_desc->numel() * infiniSizeOf(temp_out_desc->dtype());

    // Rearrange temp_out into out
    //      out: [seq_len, n_q_head, head_dim]
    //      temp_out: [n_kv_head, n_group * seq_len, head_dim] -> [n_q_head, seq_len, head_dim] -> [seq_len, n_q_head, head_dim]
    temp_out_desc = temp_out_desc->dimSplit(1, {n_group, seq_len});
    if (!temp_out_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    temp_out_desc = temp_out_desc->dimMerge(0, 1);
    if (!temp_out_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    temp_out_desc = temp_out_desc->dimPermute({1, 0, 2});
    if (!temp_out_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    infiniopRearrangeDescriptor_t rearrange_desc_out;
    CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &rearrange_desc_out, out_desc, temp_out_desc));

    // workspace size
    uint64_t workspace_size = rearranged_q_size + std::max(std::max(matmul1_workspace_size + matmul1_tensor_size, matmul1_tensor_size + softmax_workspace_size), matmul1_tensor_size + matmul2_workspace_size + matmul2_tensor_size);

    // k_cache_offset
    uint64_t k_cache_offset = 0;
    if (pos > 0) {
        k_cache_offset = pos * k_cache_desc->getByteStrides()[1];
    }

    // v_cache_offset
    uint64_t v_cache_offset = 0;
    if (pos > 0) {
        v_cache_offset = pos * v_cache_desc->getByteStrides()[1];
    }

    // create attention descriptor
    *(_infiniopAttentionDescriptor_t *)desc_ptr = new _infiniopAttentionDescriptor{
        handle->device,
        rearrange_desc_k,
        rearrange_desc_v,
        rearrange_desc_q,
        rearrange_desc_out,
        matmul1_desc,
        matmul2_desc,
        softmax_desc,
        workspace_size,
        rearranged_q_size,
        matmul1_workspace_size,
        matmul1_tensor_size,
        matmul2_workspace_size,
        matmul2_tensor_size,
        softmax_workspace_size,
        k_cache_offset,
        v_cache_offset,
    };

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopGetAttentionWorkspaceSize(infiniopAttentionDescriptor_t desc, uint64_t *size) {
    *size = ((_infiniopAttentionDescriptor_t)desc)->workspace_size;
    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopAttention(infiniopAttentionDescriptor_t desc,
                                              void *workspace,
                                              uint64_t workspace_size,
                                              void *out,
                                              void const *q,
                                              void const *k,
                                              void const *v,
                                              void *k_cache,
                                              void *v_cache,
                                              void *stream) {
    auto _desc = (_infiniopAttentionDescriptor_t)desc;
    void *_workspace = workspace;
    if (workspace_size < _desc->workspace_size) {
        return INFINI_STATUS_INTERNAL_ERROR; // STATUS_MEMORY_NOT_ALLOCATED
    }

    // concat k and v to k_cache and v_cache
    CHECK_STATUS(infiniopRearrange(_desc->rearrange_desc_k,
                                   (char *)k_cache + _desc->k_cache_offset, k, stream));

    CHECK_STATUS(infiniopRearrange(_desc->rearrange_desc_v,
                                   (char *)v_cache + _desc->v_cache_offset, v, stream));

    // rearrange q into contiguous
    void const *_q = q;
    if (_desc->rearrange_desc_q) {
        CHECK_STATUS(infiniopRearrange(_desc->rearrange_desc_q, (char *)_workspace, q, stream));
        _q = _workspace;
        _workspace = (char *)_workspace + _desc->rearranged_q_size;
    }

    // matmul1: q * full_k
    CHECK_STATUS(infiniopGemm(_desc->matmul_desc1,
                              (char *)_workspace + _desc->matmul1_tensor_size, workspace_size - _desc->matmul1_tensor_size,
                              _workspace, _q, k_cache, 1.0, 0.0, stream));
    // softmax(qk)
    CHECK_STATUS(infiniopCausalSoftmax(_desc->softmax_desc,
                                       (char *)_workspace + _desc->matmul1_tensor_size, workspace_size - _desc->matmul1_tensor_size,
                                       _workspace, _workspace, stream));
    // matmul2: softmax(qk) * full_v
    CHECK_STATUS(infiniopGemm(_desc->matmul_desc2,
                              (char *)_workspace + _desc->matmul1_tensor_size + _desc->matmul2_tensor_size,
                              workspace_size - _desc->matmul1_tensor_size - _desc->matmul2_tensor_size,
                              (char *)_workspace + _desc->matmul1_tensor_size, _workspace, v_cache, 1.0, 0.0, stream));
    // rearrange out
    CHECK_STATUS(infiniopRearrange(_desc->rearrange_desc_out, out, (char *)_workspace + _desc->matmul1_tensor_size, stream));

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopDestroyAttentionDescriptor(infiniopAttentionDescriptor_t desc) {
    if (((_infiniopAttentionDescriptor_t)desc)->rearrange_desc_q) {
        CHECK_STATUS(infiniopDestroyRearrangeDescriptor(((_infiniopAttentionDescriptor_t)desc)->rearrange_desc_q));
    }
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(((_infiniopAttentionDescriptor_t)desc)->rearrange_desc_k));
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(((_infiniopAttentionDescriptor_t)desc)->rearrange_desc_v));
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(((_infiniopAttentionDescriptor_t)desc)->rearrange_desc_out));
    CHECK_STATUS(infiniopDestroyGemmDescriptor(((_infiniopAttentionDescriptor_t)desc)->matmul_desc1));
    CHECK_STATUS(infiniopDestroyGemmDescriptor(((_infiniopAttentionDescriptor_t)desc)->matmul_desc2));
    CHECK_STATUS(infiniopDestroyCausalSoftmaxDescriptor(((_infiniopAttentionDescriptor_t)desc)->softmax_desc));
    delete (_infiniopAttentionDescriptor_t)desc;

    return INFINI_STATUS_SUCCESS;
}
