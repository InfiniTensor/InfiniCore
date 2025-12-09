#pragma once

#include "module.hpp"
#include "../context/context.hpp"
#include "../tensor.hpp"
#include <memory>
#include <optional>

namespace infinicore {

/**
 * @brief RoPE algorithm type
 *
 * Defines the algorithm variant for Rotary Position Embedding (RoPE).
 * This enum is used to select which RoPE variant to use.
 */
enum class RoPEAlgo {
    GPT_J,      ///< GPT-J style: pairs are (2j, 2j+1) for cache entry j
    GPT_NEOX,   ///< GPT-NeoX style: pairs are (j, j+head_dim/2) for cache entry j
};

} // namespace infinicore

namespace infinicore::nn {


class RoPE : public Module {
public:
    /**
     * @brief Construct a RoPE layer
     *
     * @param head_dim Dimension of each attention head (must be even)
     * @param max_seq_len Maximum sequence length for pre-computed cache
     * @param theta Base frequency for rotary embeddings (default: 10000.0)
     * @param algo RoPE algorithm type (default: RoPEAlgo::GPT_J)
     * @param dtype Data type for sin/cos cache (default: DataType::F32)
     * @param device Device to create the cache on
     */
    RoPE(size_t head_dim,
         size_t max_seq_len,
         double theta = 10000.0,
         RoPEAlgo algo = RoPEAlgo::GPT_J,
         const DataType &dtype = DataType::F32,
         const Device &device = Device());

    /**
     * @brief Forward pass: apply RoPE to a tensor
     *
     * @param x Input tensor of shape (..., head_dim) where ... is any number of dimensions
     * @param pos Position IDs tensor of shape (*,) typically [seq_len] or [batch, seq_len]
     * @return Rotated tensor with same shape as input
     *
     * Applies rotary position embeddings to the input tensor.
     * For attention mechanisms, call this method separately for query and key tensors.
     *
     * Common input shapes:
     *   - [batch, num_heads, seq_len, head_dim]
     *   - [batch, seq_len, num_heads, head_dim]
     *   - [seq_len, head_dim]
     */
    Tensor forward(const Tensor &x, const Tensor &pos) const;

    // Module information
    size_t head_dim() const { return head_dim_; }
    size_t max_seq_len() const { return max_seq_len_; }
    double theta() const { return theta_; }
    RoPEAlgo algo() const { return algo_; }
    DataType dtype() const { return dtype_; }

    // String representation
    std::string extra_repr() const;

protected:
    // Device-specific buffers (sin and cos cache tables)
    // TODO: Consider sharing cache tables among multiple RoPE layers with identical configuration
    //       to reduce memory usage in models with multiple layers using the same RoPE parameters.
    std::optional<Tensor> sin_cache_;
    std::optional<Tensor> cos_cache_;

private:
    void initialize_cache();

    size_t head_dim_;      // Dimension of each attention head
    size_t max_seq_len_;   // Maximum sequence length
    double theta_;         // Base frequency for rotary embeddings
    RoPEAlgo algo_;        // RoPE algorithm type
    DataType dtype_;       // Data type for cache tables
};

} // namespace infinicore::nn
