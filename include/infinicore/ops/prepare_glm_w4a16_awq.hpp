#pragma once

#include "common/op.hpp"

namespace infinicore::op {
void prepare_glm_w4a16_awq_(Tensor qweight, Tensor qzeros, Tensor scales,
                            const Tensor &checkpoint_weight,
                            const Tensor &channel_scales);
} // namespace infinicore::op
