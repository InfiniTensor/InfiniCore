#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <utility>

namespace infinicore::op {
class AddRMSNorm {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, float);
    static void execute(Tensor y, Tensor residual_out, Tensor a, Tensor b, Tensor weight, float epsilon = 1e-5f);
    static common::OpDispatcher<schema> &dispatcher();
};

// Fused Add and RMS Normalization
// Returns: (normalized_result, add_result)
// The add_result can be used as residual for subsequent layers
std::pair<Tensor, Tensor> add_rms_norm(Tensor a, Tensor b, Tensor weight, float epsilon = 1e-5f);
void add_rms_norm_(Tensor y, Tensor residual_out, Tensor a, Tensor b, Tensor weight, float epsilon = 1e-5f);
} // namespace infinicore::op
