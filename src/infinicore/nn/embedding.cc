#include "infinicore/nn/embedding.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinicore::nn {

Embedding::Embedding(size_t num_embeddings,
                     size_t embedding_dim,
                     std::optional<int64_t> padding_idx,
                     const DataType &dtype,
                     const Device &device)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      padding_idx_(padding_idx),
      dtype_(dtype) {

    device_ = device;

    // Validate padding_idx
    if (padding_idx_.has_value()) {
        int64_t idx = padding_idx_.value();
        if (idx < 0 || idx >= static_cast<int64_t>(num_embeddings)) {
            throw std::invalid_argument(
                "padding_idx must be within num_embeddings range, got " + std::to_string(idx) + " for num_embeddings=" + std::to_string(num_embeddings));
        }
    }

    // Initialize parameter using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_embeddings, embedding_dim}, dtype_, device));

    // If padding_idx is specified, initialize that row to zeros
    if (padding_idx_.has_value()) {
        // TODO: Set weight[padding_idx] to zeros
        // This would require a slice operation
    }

    SPDLOG_DEBUG("Created Embedding module: num_embeddings={}, embedding_dim={}, dtype={}, padding_idx={}",
                 num_embeddings, embedding_dim, static_cast<int>(dtype_),
                 padding_idx_.has_value() ? std::to_string(padding_idx_.value()) : "None");
}

Tensor Embedding::forward(const Tensor &indices) const {
    // Ensure indices are on the same device as weight
    // This avoids synchronous memcpy in ops layer which would hurt performance
    Tensor indices_on_device = indices;
    if (indices->device() != device_) {
        indices_on_device = indices->to(device_);
    }

    // Ensure indices are contiguous for efficient access
    // op::embedding now supports device-side input for graph recording
    Tensor indices_contiguous = indices_on_device->is_contiguous() ? indices_on_device : indices_on_device->contiguous();

    // Use op::embedding which now supports device-side input and batch dimension
    // This enables full graph recording support without synchronization
    return op::embedding(indices_contiguous, weight_);
}

std::string Embedding::extra_repr() const {
    std::string repr = "Embedding(num_embeddings=" + std::to_string(num_embeddings_) + ", embedding_dim=" + std::to_string(embedding_dim_) + ", dtype=" + std::to_string(static_cast<int>(dtype_));
    if (padding_idx_.has_value()) {
        repr += ", padding_idx=" + std::to_string(padding_idx_.value());
    }
    repr += ")";
    return repr;
}

} // namespace infinicore::nn
