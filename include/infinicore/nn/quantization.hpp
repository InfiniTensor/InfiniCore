// quant.hpp
#pragma once

namespace infinicore::nn {

enum class QuantScheme {
    NONE,
    COMPRESSED_TENSOR_W8A8I8,
    // 可扩展
};

struct QuantConfig {
    QuantScheme quant_scheme = QuantScheme::NONE;

    // 默认构造即“未量化”
    QuantConfig() = default;
    constexpr QuantConfig(QuantScheme scheme) : quant_scheme(scheme) {}

    QuantScheme get_quant_scheme() const { return quant_scheme; }
};

} // namespace infinicore::nn