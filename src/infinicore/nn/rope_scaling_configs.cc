#include "infinicore/nn/rope_scaling_configs.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::nn {

namespace {
// Define a portable PI constant to avoid relying on the non-standard M_PI macro
// which is missing on MSVC (Windows) by default.
constexpr float kPi = 3.14159265358979323846f;
} // anonymous namespace

// LongRopeScalingConfig Implementation
LongRopeScalingConfig::LongRopeScalingConfig(
    std::vector<float> short_factor,
    std::vector<float> long_factor,
    size_t original_max_position_embeddings,
    float factor)
    : short_factor_(std::move(short_factor)),
      long_factor_(std::move(long_factor)),
      original_max_position_embeddings_(original_max_position_embeddings),
      factor_(factor == 1.0f ? 1.0f : std::sqrt(1 + std::log(factor) / std::log(original_max_position_embeddings))) {}

float LongRopeScalingConfig::get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
    float _ext = (pos < original_max_position_embeddings_) ? short_factor_[dim_idx] : long_factor_[dim_idx];
    // The base inv_freq is multiplied by this scale.
    // Original: inv_freq = 1.0f / (_ext * pow(theta, 2j/head_dim))
    // New: inv_freq = base_inv_freq * (1.0f / _ext)
    return 1.0f / _ext;
}

float LongRopeScalingConfig::get_magnitude_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
    return factor_;
}

// Llama3RopeScalingConfig Implementation
Llama3RopeScalingConfig::Llama3RopeScalingConfig(
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    size_t original_max_position_embeddings)
    : factor_(factor),
      low_freq_factor_(low_freq_factor),
      high_freq_factor_(high_freq_factor),
      original_max_position_embeddings_(original_max_position_embeddings) {}

float Llama3RopeScalingConfig::get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
    // Calculate the wavelength corresponding to the current inverse frequency
    float wavelen = 2.0f * static_cast<float>(kPi) / base_inv_freq;

    // Compute the wavelength thresholds that separate high, mid, and low frequencies
    float low_freq_wavelen = static_cast<float>(original_max_position_embeddings_) / low_freq_factor_;
    float high_freq_wavelen = static_cast<float>(original_max_position_embeddings_) / high_freq_factor_;

    float scale = 1.0f;

    if (wavelen < low_freq_wavelen) {
        // High-frequency band: short wavelengths retain the original scale
        scale = 1.0f;
    } else if (wavelen > high_freq_wavelen) {
        // Low-frequency band: long wavelengths are directly scaled by the factor
        scale = factor_;
    } else {
        // Mid-frequency band: apply smooth linear interpolation between 1.0 and factor_
        float smooth = (static_cast<float>(original_max_position_embeddings_) / wavelen - low_freq_factor_) / (high_freq_factor_ - low_freq_factor_);
        scale = 1.0f - smooth + smooth * factor_;
    }

    // The framework applies the scale multiplicatively (inv_freq = base_inv_freq * return_value).
    // Since the Llama3 logic divides the frequency (inv_freq = base_inv_freq / scale),
    // we return the inverse of the computed scale.
    return 1.0f / scale;
}

} // namespace infinicore::nn
