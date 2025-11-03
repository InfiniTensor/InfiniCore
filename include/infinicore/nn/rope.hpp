#pragma once

#include "module.hpp"
#include "../context/context.hpp"
#include "../tensor.hpp"
#include <infiniop.h>
#include <memory>

namespace infinicore::nn {


class RoPE : public Module {
public:
    /**
     * @brief Construct a RoPE layer
     *
     * @param head_dim Dimension of each attention head (must be even)
     * @param max_seq_len Maximum sequence length for pre-computed cache
     * @param theta Base frequency for rotary embeddings (default: 10000.0)
     * @param algo RoPE algorithm type (default: INFINIOP_ROPE_ALGO_GPT_J)
     * @param dtype Data type for sin/cos cache (default: DataType::F32)
     * @param device Device to create the cache on
     */
    RoPE(size_t head_dim,
         size_t max_seq_len,
         double theta = 10000.0,
         infiniopRoPEAlgo_t algo = INFINIOP_ROPE_ALGO_GPT_J,
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
    infiniopRoPEAlgo_t algo() const { return algo_; }
    DataType dtype() const { return dtype_; }

    // String representation
    std::string extra_repr() const;

    // Accessors for parameters
    Tensor sin_cache() const { return sin_cache_; }
    Tensor cos_cache() const { return cos_cache_; }

protected:
    // Parameters (sin and cos cache tables)
    INFINICORE_NN_PARAMETER(sin_cache);
    INFINICORE_NN_PARAMETER(cos_cache);

private:
    void initialize_cache();

    size_t head_dim_;      // Dimension of each attention head
    size_t max_seq_len_;   // Maximum sequence length
    double theta_;         // Base frequency for rotary embeddings
    infiniopRoPEAlgo_t algo_;  // RoPE algorithm type
    DataType dtype_;       // Data type for cache tables
};

} // namespace infinicore::nn
