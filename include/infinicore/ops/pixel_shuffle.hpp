#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class PixelShuffle {
public:
    using schema = void (*)(Tensor, Tensor, int64_t);
    static void execute(Tensor output, Tensor input, int64_t upscale_factor);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor pixel_shuffle(Tensor input, int64_t upscale_factor);
void pixel_shuffle_(Tensor output, Tensor input, int64_t upscale_factor);

} // namespace infinicore::op

