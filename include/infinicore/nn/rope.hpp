#pragma once

#include "module.hpp"
#include "../context/context.hpp"
#include "../tensor.hpp"
#include <memory>

namespace infinicore::nn {

class RoPE : public Module {
public:
    /**
     * @brief RoPE rotation algorithm type (kernel-level and frequency generation)
     */
    enum class Algo {
        GPT_J = 0,    // GPT-J style: pairs dimensions as (2j, 2j+1), frequency for cache entry j is theta^(-2j/head_dim)
        GPT_NEOX = 1, // GPT-NeoX style: pairs dimensions as (j, j+head_dim/2), frequency for cache entry j is theta^(-j/head_dim)
    };

    /**
     * @brief Construct a RoPE layer
     *
     * @param head_dim Dimension of each attention head (must be even)
     * @param max_seq_len Maximum sequence length for pre-computed cache
     * @param theta Base frequency for rotary embeddings (default: 10000.0)
     * @param freq_gen Frequency generation method (default: Algo::GPT_J)
     * @param algo Rotation algorithm type for kernel (default: Algo::GPT_J)
     * @param dtype Data type for sin/cos cache (default: DataType::F32)
     * @param device Device to create the cache on
     */
    RoPE(size_t head_dim,
         size_t max_seq_len,
         double theta = 10000.0,
         Algo freq_gen = Algo::GPT_J,
         Algo algo = Algo::GPT_J,
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
    Algo freq_gen() const { return freq_gen_; }
    Algo algo() const { return algo_; }
    DataType dtype() const { return dtype_; }

    // String representation
    std::string extra_repr() const;

protected:
    // Buffers (sin and cos cache tables) - not exposed in state_dict
    INFINICORE_NN_BUFFER(sin_cache);
    INFINICORE_NN_BUFFER(cos_cache);

private:
    void initialize_cache();

    size_t head_dim_;      // Dimension of each attention head
    size_t max_seq_len_;   // Maximum sequence length
    double theta_;         // Base frequency for rotary embeddings
    Algo freq_gen_;     // Frequency generation method
    Algo algo_;            // Rotation algorithm type (kernel-level)
    DataType dtype_;       // Data type for cache tables
};

} // namespace infinicore::nn
