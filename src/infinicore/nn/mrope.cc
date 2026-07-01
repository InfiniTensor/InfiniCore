#include "infinicore/nn/mrope.hpp"
#include "../../utils.h"
#include "../utils.hpp"
#include "infinicore/ops/mrope.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace infinicore::nn {

MRoPE::MRoPE(size_t head_dim,
             size_t rotary_dim,
             size_t max_seq_len,
             double theta,
             std::array<int, 3> section,
             bool interleaved,
             const DataType &dtype,
             const Device &device)
    : head_dim_(head_dim),
      rotary_dim_(rotary_dim),
      max_seq_len_(max_seq_len),
      theta_(theta),
      section_(section),
      interleaved_(interleaved),
      dtype_(dtype) {
    if (rotary_dim_ % 2 != 0) {
        throw std::invalid_argument("rotary_dim must be even for MRoPE, got " + std::to_string(rotary_dim_));
    }
    if (rotary_dim_ == 0 || rotary_dim_ > head_dim_) {
        throw std::invalid_argument("rotary_dim must be in (0, head_dim] for MRoPE");
    }
    if (2 * static_cast<size_t>(section_[0] + section_[1] + section_[2]) != rotary_dim_) {
        throw std::invalid_argument("MRoPE section sum must equal rotary_dim / 2");
    }
    device_ = device;
    initialize_cache();
}

void MRoPE::initialize_cache() {
    const size_t cache_dim = rotary_dim_ / 2;
    const size_t numel = max_seq_len_ * cache_dim;
    INFINICORE_NN_BUFFER_INIT(sin_cache, ({max_seq_len_, cache_dim}, dtype_, device_));
    INFINICORE_NN_BUFFER_INIT(cos_cache, ({max_seq_len_, cache_dim}, dtype_, device_));

    std::vector<float> sin_data(numel);
    std::vector<float> cos_data(numel);
    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        for (size_t dim_idx = 0; dim_idx < cache_dim; ++dim_idx) {
            const float inv_freq = 1.0f / std::pow(static_cast<float>(theta_), 2.0f * static_cast<float>(dim_idx) / static_cast<float>(rotary_dim_));
            const float angle = static_cast<float>(pos) * inv_freq;
            const size_t offset = pos * cache_dim + dim_idx;
            sin_data[offset] = std::sin(angle);
            cos_data[offset] = std::cos(angle);
        }
    }

    const auto cpu_device = Device(Device::Type::CPU, 0);
    if (dtype_ == DataType::F32) {
        auto sin_cpu = Tensor::from_blob(sin_data.data(), {max_seq_len_, cache_dim}, DataType::F32, cpu_device);
        auto cos_cpu = Tensor::from_blob(cos_data.data(), {max_seq_len_, cache_dim}, DataType::F32, cpu_device);
        sin_cache_->copy_from(sin_cpu);
        cos_cache_->copy_from(cos_cpu);
        return;
    }
    if (dtype_ == DataType::BF16) {
        std::vector<bf16_t> sin_bf16(numel);
        std::vector<bf16_t> cos_bf16(numel);
        for (size_t i = 0; i < numel; ++i) {
            sin_bf16[i] = utils::cast<bf16_t, float>(sin_data[i]);
            cos_bf16[i] = utils::cast<bf16_t, float>(cos_data[i]);
        }
        auto sin_cpu = Tensor::from_blob(sin_bf16.data(), {max_seq_len_, cache_dim}, DataType::BF16, cpu_device);
        auto cos_cpu = Tensor::from_blob(cos_bf16.data(), {max_seq_len_, cache_dim}, DataType::BF16, cpu_device);
        sin_cache_->copy_from(sin_cpu);
        cos_cache_->copy_from(cos_cpu);
        return;
    }
    if (dtype_ == DataType::F16) {
        std::vector<fp16_t> sin_f16(numel);
        std::vector<fp16_t> cos_f16(numel);
        for (size_t i = 0; i < numel; ++i) {
            sin_f16[i] = utils::cast<fp16_t, float>(sin_data[i]);
            cos_f16[i] = utils::cast<fp16_t, float>(cos_data[i]);
        }
        auto sin_cpu = Tensor::from_blob(sin_f16.data(), {max_seq_len_, cache_dim}, DataType::F16, cpu_device);
        auto cos_cpu = Tensor::from_blob(cos_f16.data(), {max_seq_len_, cache_dim}, DataType::F16, cpu_device);
        sin_cache_->copy_from(sin_cpu);
        cos_cache_->copy_from(cos_cpu);
        return;
    }
    throw std::runtime_error("MRoPE cache dtype conversion not supported for dtype: " + std::to_string(static_cast<int>(dtype_)));
}

std::pair<Tensor, Tensor> MRoPE::forward(const Tensor &q,
                                         const Tensor &k,
                                         const Tensor &positions) const {
    const size_t num_tokens = q->size(0);
    auto q_flat = q->contiguous()->view({num_tokens, q->size(1) * head_dim_});
    auto k_flat = k->contiguous()->view({num_tokens, k->size(1) * head_dim_});
    auto q_out = Tensor::empty(q_flat->shape(), q_flat->dtype(), q_flat->device());
    auto k_out = Tensor::empty(k_flat->shape(), k_flat->dtype(), k_flat->device());
    op::mrope_(q_out,
               k_out,
               q_flat,
               k_flat,
               cos_cache_,
               sin_cache_,
               positions,
               static_cast<int>(head_dim_),
               static_cast<int>(rotary_dim_),
               section_[0],
               section_[1],
               section_[2],
               interleaved_);
    return {q_out->view(q->shape()), k_out->view(k->shape())};
}

std::pair<Tensor, Tensor> MRoPE::forward(const Tensor &q_out,
                                         const Tensor &k_out,
                                         const Tensor &q,
                                         const Tensor &k,
                                         const Tensor &positions) const {
    const size_t num_tokens = q->size(0);
    auto q_flat = q->contiguous()->view({num_tokens, q->size(1) * head_dim_});
    auto k_flat = k->contiguous()->view({num_tokens, k->size(1) * head_dim_});
    auto q_out_flat = q_out->view({num_tokens, q->size(1) * head_dim_});
    auto k_out_flat = k_out->view({num_tokens, k->size(1) * head_dim_});
    op::mrope_(q_out_flat,
               k_out_flat,
               q_flat,
               k_flat,
               cos_cache_,
               sin_cache_,
               positions,
               static_cast<int>(head_dim_),
               static_cast<int>(rotary_dim_),
               section_[0],
               section_[1],
               section_[2],
               interleaved_);
    return {q_out, k_out};
}

std::string MRoPE::extra_repr() const {
    return "MRoPE(head_dim=" + std::to_string(head_dim_)
         + ", rotary_dim=" + std::to_string(rotary_dim_)
         + ", max_seq_len=" + std::to_string(max_seq_len_)
         + ", theta=" + std::to_string(theta_)
         + ", section=[" + std::to_string(section_[0]) + "," + std::to_string(section_[1]) + "," + std::to_string(section_[2]) + "]"
         + ", interleaved=" + (interleaved_ ? "true" : "false")
         + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
