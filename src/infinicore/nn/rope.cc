#include "infinicore/nn/rope.hpp"
#include "../utils.hpp"
#include "infinicore/ops.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinicore::nn {

RoPE::RoPE(size_t head_dim,
           size_t max_seq_len,
           double theta,
           infiniopRoPEAlgo_t algo,
           const DataType &dtype,
           const Device &device)
    : head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      theta_(theta),
      algo_(algo),
      dtype_(dtype) {

    if (head_dim % 2 != 0) {
        throw std::invalid_argument("head_dim must be even for RoPE, got " + std::to_string(head_dim));
    }

    device_ = device;

    // Initialize cache tables
    initialize_cache();

    spdlog::debug("Created RoPE module: head_dim={}, max_seq_len={}, theta={}, algo={}, dtype={}",
                  head_dim, max_seq_len, theta, static_cast<int>(algo), static_cast<int>(dtype_));
}

void RoPE::initialize_cache() {
    size_t cache_dim = head_dim_ / 2;

    // Create sin and cos cache tables: [max_seq_len, cache_dim]
    INFINICORE_NN_PARAMETER_INIT(sin_cache, ({max_seq_len_, cache_dim}, dtype_, device_));
    INFINICORE_NN_PARAMETER_INIT(cos_cache, ({max_seq_len_, cache_dim}, dtype_, device_));

    // Pre-compute sin and cos values
    // The frequency calculation differs based on algorithm:
    // - GPT_J: pairs are (2j, 2j+1) for cache entry j, frequency for dimension 2j is theta^(-2j/head_dim)
    // - GPT_NEOX: pairs are (j, j+head_dim/2) for cache entry j, frequency for dimension j is theta^(-j/head_dim)

    // Compute on CPU first, then copy to device
    auto cpu_device = Device(Device::Type::CPU, 0);

    // Allocate CPU buffers
    std::vector<float> sin_data(max_seq_len_ * cache_dim);
    std::vector<float> cos_data(max_seq_len_ * cache_dim);

    for (size_t pos = 0; pos < max_seq_len_; pos++) {
        for (size_t j = 0; j < cache_dim; j++) {
            // Compute inverse frequency based on algorithm
            double inv_freq;

            if (algo_ == INFINIOP_ROPE_ALGO_GPT_J) {
                // GPT_J: pairs are (2j, 2j+1) for cache entry j
                // Frequency for pair j: theta^(-2j/head_dim)
                inv_freq = 1.0 / std::pow(theta_, 2.0 * static_cast<double>(j) / static_cast<double>(head_dim_));
            } else if (algo_ == INFINIOP_ROPE_ALGO_GPT_NEOX) {
                // GPT_NEOX: pairs are (j, j+head_dim/2) for cache entry j
                // Frequency for pair j (corresponding to dimension j): theta^(-j/head_dim)
                inv_freq = 1.0 / std::pow(theta_, static_cast<double>(j) / static_cast<double>(head_dim_));
            } else {
                throw std::runtime_error("Unsupported RoPE algorithm: " + std::to_string(static_cast<int>(algo_)));
            }

            // Compute angle: position * inverse_frequency
            double angle = static_cast<double>(pos) * inv_freq;

            // Compute sin and cos
            sin_data[pos * cache_dim + j] = static_cast<float>(std::sin(angle));
            cos_data[pos * cache_dim + j] = static_cast<float>(std::cos(angle));
        }
    }

    // Create CPU tensors and copy data
    auto sin_cpu = Tensor::from_blob(sin_data.data(), {max_seq_len_, cache_dim}, DataType::F32, cpu_device);
    auto cos_cpu = Tensor::from_blob(cos_data.data(), {max_seq_len_, cache_dim}, DataType::F32, cpu_device);

    // Copy to device
    // Note: Cache is created with dtype_, but we compute in F32 for precision.
    // If dtype_ != F32, copy_from will fail. For now, we only support F32 cache.
    // TODO: Add dtype conversion support when cast operation is available
    if (dtype_ != DataType::F32) {
        throw std::runtime_error(
            "RoPE cache dtype conversion not yet supported. Please use DataType::F32 for cache. "
            "Requested dtype: "
            + std::to_string(static_cast<int>(dtype_)));
    }

    // copy_from handles cross-device copying automatically
    // Direct copy from CPU to target device avoids double copying
    sin_cache_->copy_from(sin_cpu);
    cos_cache_->copy_from(cos_cpu);
}

Tensor RoPE::forward(const Tensor &x, const Tensor &pos) const {
    // Delegate to InfiniCore op (backed by InfiniRT/InfiniOP)
    // Validation is handled by the op layer
    return op::rope(x, pos, sin_cache_, cos_cache_, algo_);
}

std::string RoPE::extra_repr() const {
    std::string algo_str = (algo_ == INFINIOP_ROPE_ALGO_GPT_J) ? "GPT_J" : "GPT_NEOX";
    return "RoPE(head_dim=" + std::to_string(head_dim_) + ", max_seq_len=" + std::to_string(max_seq_len_) + ", theta=" + std::to_string(theta_) + ", algo=" + algo_str + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
