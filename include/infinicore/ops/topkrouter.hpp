#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class TopKRouter {
public:
    // values_output: [N, topk] float32
    // indices_output: [N, topk] int32
    // input: [N, width] float16/bfloat16/float32
    // correction_bias: [width] float32
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float, size_t);
    static void execute(Tensor values_output,
                        Tensor indices_output,
                        Tensor input,
                        Tensor correction_bias,
                        float routed_scaling_factor,
                        size_t topk);
    static common::OpDispatcher<schema> &dispatcher();
};

std::pair<Tensor, Tensor> topkrouter(Tensor input,
                                    Tensor correction_bias,
                                    float routed_scaling_factor,
                                    size_t topk);

void topkrouter_(Tensor values_output,
                 Tensor indices_output,
                 Tensor input,
                 Tensor correction_bias,
                 float routed_scaling_factor,
                 size_t topk);

} // namespace infinicore::op
