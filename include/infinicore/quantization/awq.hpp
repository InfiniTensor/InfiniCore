#pragma once
#include "base_quantization.hpp"
namespace infinicore::quantization {

class AWQ : public BaseQuantization {
    // This is a temporary class that currently only returns AWQ_W4A16.
    // Future enhancements should parse quant_config to extract detailed quantization
    // information and support multiple quantization schemes.
public:
    explicit AWQ(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {};

    infinicore::quantization::QuantScheme
    get_quant_scheme() const override {
        return infinicore::quantization::QuantScheme::AWQ_W4A16;
    };
};

} // namespace infinicore::quantization
