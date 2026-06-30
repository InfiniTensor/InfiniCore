#include "infinicore/nn/rope.hpp"
#include "../../utils.h"
#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinicore::nn {
namespace {

void sync_cache_copy() {
    infinicore::context::syncStream();
}

} // namespace

RoPE::RoPE(size_t head_dim,
           size_t rotary_dim,
           size_t max_seq_len,
           double theta,
           Algo algo,
           const DataType &dtype,
           const Device &device,
           std::shared_ptr<RopeScalingConfig> scaling)
    : rotary_dim_(rotary_dim),
      head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      theta_(theta),
      algo_(algo),
      dtype_(dtype),
      scaling_(scaling) {
    // TODO use head_dim
    if (rotary_dim % 2 != 0) {
        throw std::invalid_argument("rotary_dim must be even for RoPE, got " + std::to_string(rotary_dim));
    }
    assert((rotary_dim > 0) && (rotary_dim <= head_dim_));
    device_ = device;

    // Initialize cache tables
    initialize_cache();
}

void RoPE::initialize_cache() {
    size_t cache_dim = rotary_dim_ / 2;

    // Create sin and cos cache tables: [max_seq_len, cache_dim]
    INFINICORE_NN_BUFFER_INIT(sin_cache, ({max_seq_len_, cache_dim}, dtype_, device_));
    INFINICORE_NN_BUFFER_INIT(cos_cache, ({max_seq_len_, cache_dim}, dtype_, device_));

    // Pre-compute sin and cos values
    // Frequency generation always uses GPT-J style (theta^(-2j/rotary_dim)).
    // The rotation algorithm (algo_) controls how dimensions are paired in the kernel.

    // Compute on CPU first, then copy to device
    auto cpu_device = Device(Device::Type::CPU, 0);

    // Allocate CPU buffers
    std::vector<float> sin_data(max_seq_len_ * cache_dim);
    std::vector<float> cos_data(max_seq_len_ * cache_dim);

    for (size_t pos = 0; pos < max_seq_len_; pos++) {
        for (size_t dim_idx = 0; dim_idx < cache_dim; dim_idx++) {
            // 1. Base inverse frequency (shared across all RoPE types)
            float base_inv_freq = 1.0f / std::pow(static_cast<float>(theta_), 2.0f * static_cast<float>(dim_idx) / static_cast<float>(rotary_dim_));

            // 2. Polymorphic scaling resolution
            // Passing pre-computed base_inv_freq avoids redundant pow() calculations in subclasses
            float freq_scale = scaling_ ? scaling_->get_freq_scale(pos, dim_idx, base_inv_freq) : 1.0f;
            float mag_scale = scaling_ ? scaling_->get_magnitude_scale(pos, dim_idx, base_inv_freq) : 1.0f;

            // 3. Compute final angle and sin/cos values
            float angle = static_cast<float>(pos) * base_inv_freq * freq_scale;

            sin_data[pos * cache_dim + dim_idx] = std::sin(angle) * mag_scale;
            cos_data[pos * cache_dim + dim_idx] = std::cos(angle) * mag_scale;
        }
    }

    // Convert to target dtype on CPU (matching Python's numpy astype conversion pattern)
    // Python: np_array.astype(ml_dtypes.bfloat16, copy=True) converts F32 -> BF16
    if (dtype_ == DataType::F32) {
        // Direct use of F32 data
        auto sin_f32_cpu = Tensor::from_blob(sin_data.data(), {max_seq_len_, cache_dim}, DataType::F32, cpu_device);
        auto cos_f32_cpu = Tensor::from_blob(cos_data.data(), {max_seq_len_, cache_dim}, DataType::F32, cpu_device);
        sin_cache_->copy_from(sin_f32_cpu);
        cos_cache_->copy_from(cos_f32_cpu);
        sync_cache_copy();
    } else if (dtype_ == DataType::BF16) {
        // Convert F32 to BF16 using the same conversion as Python's ml_dtypes.bfloat16
        // This uses round-to-nearest-even (matching _f32_to_bf16 implementation)
        std::vector<bf16_t> sin_bf16_data(max_seq_len_ * cache_dim);
        std::vector<bf16_t> cos_bf16_data(max_seq_len_ * cache_dim);

        for (size_t i = 0; i < sin_data.size(); i++) {
            sin_bf16_data[i] = utils::cast<bf16_t, float>(sin_data[i]);
            cos_bf16_data[i] = utils::cast<bf16_t, float>(cos_data[i]);
        }

        auto sin_bf16_cpu = Tensor::from_blob(sin_bf16_data.data(), {max_seq_len_, cache_dim}, DataType::BF16, cpu_device);
        auto cos_bf16_cpu = Tensor::from_blob(cos_bf16_data.data(), {max_seq_len_, cache_dim}, DataType::BF16, cpu_device);

        // copy_from handles cross-device copying to target device
        sin_cache_->copy_from(sin_bf16_cpu);
        cos_cache_->copy_from(cos_bf16_cpu);
        sync_cache_copy();
    } else if (dtype_ == DataType::F16) {
        // Convert F32 to F16
        std::vector<fp16_t> sin_f16_data(max_seq_len_ * cache_dim);
        std::vector<fp16_t> cos_f16_data(max_seq_len_ * cache_dim);

        for (size_t i = 0; i < sin_data.size(); i++) {
            sin_f16_data[i] = utils::cast<fp16_t, float>(sin_data[i]);
            cos_f16_data[i] = utils::cast<fp16_t, float>(cos_data[i]);
        }

        auto sin_f16_cpu = Tensor::from_blob(sin_f16_data.data(), {max_seq_len_, cache_dim}, DataType::F16, cpu_device);
        auto cos_f16_cpu = Tensor::from_blob(cos_f16_data.data(), {max_seq_len_, cache_dim}, DataType::F16, cpu_device);

        sin_cache_->copy_from(sin_f16_cpu);
        cos_cache_->copy_from(cos_f16_cpu);
        sync_cache_copy();
    } else {
        throw std::runtime_error(
            "RoPE cache dtype conversion not yet supported for dtype: "
            + std::to_string(static_cast<int>(dtype_)));
    }
}

Tensor RoPE::forward(const Tensor &x, const Tensor &pos, bool in_place) const {
    Tensor y;
    if (in_place) {
        y = Tensor(x);
    } else {
        y = Tensor::empty(x->shape(), x->dtype(), x->device());
        if (rotary_dim_ < head_dim_) {
            y->copy_from(x);
        }
    }

    size_t ndim = x->ndim();
    op::rope_(y->narrow({{ndim - 1, 0, rotary_dim_}}),
              x->narrow({{ndim - 1, 0, rotary_dim_}}),
              pos, sin_cache_, cos_cache_, algo_);
    return y;
}

Tensor RoPE::forward(const Tensor &y, const Tensor &x, const Tensor &pos) const {
    size_t ndim = x->ndim();
    op::rope_(y->narrow({{ndim - 1, 0, rotary_dim_}}),
              x->narrow({{ndim - 1, 0, rotary_dim_}}),
              pos, sin_cache_, cos_cache_, algo_);
    return y;
}

std::string RoPE::extra_repr() const {
    std::string algo_str = (algo_ == Algo::GPT_J) ? "GPT_J" : "GPT_NEOX";
    return "RoPE(head_dim=" + std::to_string(head_dim_) + ", rotary_dim=" + std::to_string(rotary_dim_) + ", max_seq_len=" + std::to_string(max_seq_len_) + ", theta=" + std::to_string(theta_) + ", algo=" + algo_str + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
