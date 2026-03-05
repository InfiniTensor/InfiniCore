#pragma once
#include "nlohmann/json.hpp"
#include "quantization_scheme.hpp"

namespace infinicore::quantization {
class BaseQuantization {
    // Base class for quantization schemes. Intended to be extended to support various quantization methods.
public:
    explicit BaseQuantization(const nlohmann::json &quant_config) : quant_config_(quant_config) {};
    virtual ~BaseQuantization() = default;

    virtual infinicore::quantization::QuantScheme get_quant_scheme() const = 0;

protected:
    nlohmann::json quant_config_;
};
} // namespace infinicore::quantization
