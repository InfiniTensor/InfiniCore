#pragma once

#include "../device.hpp"
#include "../tensor.hpp"

namespace infinicore::op {

void dynamic_scaled_int8_quant_(Tensor output, const Tensor &input, Tensor input_scales);
Tensor dynamic_scaled_int8_quant(const Tensor &input, Tensor input_scales);

} // namespace infinicore::op
