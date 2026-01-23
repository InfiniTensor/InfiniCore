#pragma once

#include "module.hpp"
#include "../ops.hpp"

namespace infinicore::nn {

/**
 * @brief Layer Normalization
 *
 * Applies LayerNorm over the last dimension.
 *
 * Formula: y = (x - mean) / sqrt(var + eps) * weight + bias
 */
class LayerNorm : public Module {
public:
    /**
     * @brief Construct a LayerNorm layer
     *
     * @param normalized_shape Size of the feature dimension to normalize (typically hidden_size)
     * @param eps Small constant for numerical stability (default: 1e-5)
     * @param dtype Data type for the weight/bias (default: DataType::F32)
     * @param device Device to create the parameters on
     */
    LayerNorm(size_t normalized_shape,
              double eps = 1e-5,
              const DataType &dtype = DataType::F32,
              const Device &device = Device());

    /**
     * @brief Forward pass: apply LayerNorm
     *
     * @param x Input tensor of shape (*, normalized_shape)
     * @return Normalized tensor with same shape as input
     */
    Tensor forward(const Tensor &x) const;

    // Module information
    size_t normalized_shape() const { return normalized_shape_; }
    double eps() const { return eps_; }
    DataType dtype() const { return dtype_; }

    // String representation
    std::string extra_repr() const;

    // Accessors for parameters
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }

protected:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);

private:
    size_t normalized_shape_;
    double eps_;
    DataType dtype_;
};

} // namespace infinicore::nn
