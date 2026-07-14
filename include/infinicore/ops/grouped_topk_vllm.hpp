#pragma once
#include "../device.hpp"
#include "../tensor.hpp"
#include <cstdint>
#include <string>
namespace infinicore::op {
void grouped_topk_vllm_(Tensor topk_weights, Tensor topk_ids, const Tensor &scores, int64_t num_expert_group, int64_t topk_group, bool renormalize, float routed_scaling_factor, const Tensor &bias = Tensor(), const std::string &scoring_func = "softmax");
}
