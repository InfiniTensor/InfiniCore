// quant.hpp
#pragma once

namespace infinicore::nn {

enum class QuantType {
    NONE,
    COMPRESSED_TENSOR
    // 可扩展
};

struct QuantConfig {
    QuantType quant_type = QuantType::NONE;
    // bool use_zero_point = false;
    // bool per_channel = true;
    // int group_size = -1; // -1 表示无分组（如 per-channel）

    // 默认构造即“未量化”
    QuantConfig() = default;
    constexpr QuantConfig(QuantType type) : quant_type(type) {}

    QuantType get_quant_type() const { return quant_type; }
};

} // namespace infinicore::nn