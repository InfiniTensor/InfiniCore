#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

Tensor pixel_shuffle(Tensor input, int64_t upscale_factor);

} // namespace infinicore::op

