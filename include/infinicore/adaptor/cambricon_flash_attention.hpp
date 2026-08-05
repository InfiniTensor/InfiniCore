#pragma once

#if defined(ENABLE_CAMBRICON_API) && defined(ENABLE_FLASH_ATTN)

#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

std::vector<at::Tensor>
mha_fwd(const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        std::optional<at::Tensor> &out,
        std::optional<at::Tensor> &alibi_slopes,
        float dropout_p,
        float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool return_softmax,
        std::optional<at::Generator> gen);

std::vector<at::Tensor>
mha_varlen_fwd(const at::Tensor &q,
               const at::Tensor &k,
               const at::Tensor &v,
               std::optional<at::Tensor> &out,
               const at::Tensor &cu_seqlens_q,
               const at::Tensor &cu_seqlens_k,
               std::optional<at::Tensor> &seqused_k,
               std::optional<at::Tensor> &alibi_slopes,
               int max_seqlen_q,
               int max_seqlen_k,
               float dropout_p,
               float softmax_scale,
               bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               bool return_softmax,
               std::optional<at::Generator> gen);

namespace infinicore::adaptor::cambricon_flash_attn {

struct PackedCache {
    at::Tensor key;
    at::Tensor value;
    int max_seqlen;
};

inline std::vector<int32_t> to_host_i32(const at::Tensor &tensor) {
    auto host = tensor.to(at::kCPU).to(at::kInt).contiguous();
    const auto *data = host.data_ptr<int32_t>();
    return {data, data + host.numel()};
}

inline std::vector<int32_t>
lengths_from_cumulative(const at::Tensor &cu_seqlens) {
    auto cumulative = to_host_i32(cu_seqlens);
    if (cumulative.size() < 2 || cumulative.front() != 0) {
        throw std::runtime_error(
            "Cambricon flash-attn expects cumulative sequence lengths starting at zero");
    }
    std::vector<int32_t> lengths(cumulative.size() - 1);
    for (size_t i = 0; i < lengths.size(); ++i) {
        lengths[i] = cumulative[i + 1] - cumulative[i];
        if (lengths[i] < 0) {
            throw std::runtime_error(
                "Cambricon flash-attn sequence lengths must be nondecreasing");
        }
    }
    return lengths;
}

inline PackedCache gather_paged_cache(
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &block_table,
    const std::vector<int32_t> &lengths) {
    if (block_table.dim() != 2
        || static_cast<size_t>(block_table.size(0)) != lengths.size()) {
        throw std::runtime_error(
            "Cambricon flash-attn block table shape does not match the batch");
    }
    auto table_host = block_table.to(at::kCPU).to(at::kInt).contiguous();
    const auto *table = table_host.data_ptr<int32_t>();
    const int64_t table_width = table_host.size(1);
    const int64_t block_size = key_cache.size(1);
    std::vector<int64_t> block_ids;
    std::vector<int64_t> offsets;
    int max_seqlen = 0;
    int64_t total_tokens = 0;
    for (size_t batch = 0; batch < lengths.size(); ++batch) {
        const int32_t length = lengths[batch];
        if ((length + block_size - 1) / block_size > table_width) {
            throw std::runtime_error(
                "Cambricon flash-attn block table is too short");
        }
        max_seqlen = std::max(max_seqlen, static_cast<int>(length));
        total_tokens += length;
        for (int32_t token = 0; token < length; ++token) {
            block_ids.push_back(
                table[batch * table_width + token / block_size]);
            offsets.push_back(token % block_size);
        }
    }
    auto index_options = block_table.options().dtype(at::kLong);
    auto blocks = at::tensor(block_ids, index_options);
    auto tokens = at::tensor(offsets, index_options);
    using at::indexing::Slice;
    auto key = key_cache.index({blocks, tokens, Slice(), Slice()});
    auto value = value_cache.index({blocks, tokens, Slice(), Slice()});
    if (key.size(0) != total_tokens || value.size(0) != total_tokens) {
        throw std::runtime_error(
            "Cambricon flash-attn failed to gather the paged KV cache");
    }
    return {key.contiguous(), value.contiguous(), max_seqlen};
}

} // namespace infinicore::adaptor::cambricon_flash_attn

#endif
