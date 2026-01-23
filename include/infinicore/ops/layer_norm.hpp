#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class LayerNorm {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float);
    static void execute(Tensor output,
                        Tensor input_standardization,
                        Tensor input_std_deviation,
                        Tensor input,
                        Tensor weight,
                        Tensor bias,
                        float epsilon);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor layer_norm(Tensor input, Tensor weight, Tensor bias, float epsilon = 1e-5f);
void layer_norm_(Tensor output,
                 Tensor input_standardization,
                 Tensor input_std_deviation,
                 Tensor input,
                 Tensor weight,
                 Tensor bias,
                 float epsilon = 1e-5f);
} // namespace infinicore::op
