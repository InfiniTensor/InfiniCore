#pragma once

#include "../../../utils.h"
#include "../../tensor.h"
#include <limits>

namespace op::mamba2_scan {

struct Mamba2ScanInfo {
    infiniDtype_t dtype;
    size_t tokens, heads, head_dim, groups, state_size, pool_size, requests;
    static constexpr size_t chunk_size = 256;

    bool single_chunk() const { return tokens <= chunk_size || tokens == requests; }
    size_t max_chunks() const { return (tokens + chunk_size - 1) / chunk_size; }
    size_t chunk_slots() const { return max_chunks() + requests; }
    size_t state_elements() const { return heads * head_dim * state_size; }
    size_t workspace_bytes() const {
        if (single_chunk()) {
            return 0;
        }
        return sizeof(float) * (2 * tokens * heads + chunk_slots() * (state_elements() + heads));
    }

    static utils::Result<Mamba2ScanInfo> create(
        infiniopTensorDescriptor_t out, infiniopTensorDescriptor_t x,
        infiniopTensorDescriptor_t dt, infiniopTensorDescriptor_t b,
        infiniopTensorDescriptor_t c, infiniopTensorDescriptor_t a,
        infiniopTensorDescriptor_t d, infiniopTensorDescriptor_t dt_bias,
        infiniopTensorDescriptor_t state, infiniopTensorDescriptor_t offsets,
        infiniopTensorDescriptor_t initial_indices, infiniopTensorDescriptor_t final_indices) {
        for (auto tensor : {out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices}) {
            if (tensor == nullptr) {
                return INFINI_STATUS_NULL_POINTER;
            }
            if (!tensor->isContiguous()) {
                return INFINI_STATUS_BAD_TENSOR_STRIDES;
            }
        }
        const auto dtype = x->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        for (auto tensor : {out, dt, b, c}) {
            if (tensor->dtype() != dtype) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        }
        for (auto tensor : {a, d, dt_bias, state}) {
            if (tensor->dtype() != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        }
        for (auto tensor : {offsets, initial_indices, final_indices}) {
            if (tensor->dtype() != INFINI_DTYPE_I32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            if (tensor->ndim() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
        if (x->ndim() != 3 || dt->ndim() != 2 || b->ndim() != 3 || state->ndim() != 4
            || offsets->dim(0) < 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t tokens = x->dim(0), heads = x->dim(1), head_dim = x->dim(2);
        const size_t groups = b->dim(1), state_size = b->dim(2);
        const size_t requests = offsets->dim(0) - 1;
        if (tokens == 0 || heads == 0 || head_dim == 0 || groups == 0 || state_size == 0
            || state_size > 256 || heads % groups != 0 || state->dim(0) < 2
            || requests > tokens || requests >= state->dim(0)
            || tokens > static_cast<size_t>(std::numeric_limits<int32_t>::max())
            || requests > 65535 || (tokens + chunk_size - 1) / chunk_size > 65535) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (out->shape() != x->shape() || c->shape() != b->shape() || b->dim(0) != tokens
            || dt->shape() != std::vector<size_t>{tokens, heads}
            || a->shape() != std::vector<size_t>{heads} || d->shape() != a->shape()
            || dt_bias->shape() != a->shape()
            || state->shape() != std::vector<size_t>{state->dim(0), heads, head_dim, state_size}
            || initial_indices->shape() != std::vector<size_t>{requests}
            || final_indices->shape() != initial_indices->shape()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        // Bound address arithmetic and launch dimensions before multiplying sizes.
        const size_t max_elements = std::numeric_limits<size_t>::max() / sizeof(float);
        if (heads > max_elements / head_dim
            || heads * head_dim > max_elements / state_size
            || heads * head_dim > max_elements / tokens
            || groups > max_elements / state_size / tokens) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        Mamba2ScanInfo info{dtype, tokens, heads, head_dim, groups, state_size, state->dim(0), requests};
        const size_t state_elements = info.state_elements();
        const size_t max_grid_x = std::numeric_limits<int32_t>::max();
        if (state_elements > max_elements / info.pool_size
            || heads > max_grid_x / ((head_dim - 1) / 4 + 1)
            || (tokens * heads - 1) / 256 + 1 > max_grid_x
            || (state_elements - 1) / 256 + 1 > max_grid_x) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (!info.single_chunk()
            && (tokens > max_elements / heads / 2
                || info.chunk_slots() > (max_elements - 2 * tokens * heads) / (state_elements + heads))) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        return utils::Result<Mamba2ScanInfo>(info);
    }
};

} // namespace op::mamba2_scan
