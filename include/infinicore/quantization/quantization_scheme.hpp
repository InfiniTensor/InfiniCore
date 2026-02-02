// quant.hpp
#pragma once

namespace infinicore::quantization {

enum class QuantScheme {
    NONE,
    COMPRESSED_TENSOR_W8A8I8,
    AWQ_W4A16,
};

} // namespace infinicore::quantization
