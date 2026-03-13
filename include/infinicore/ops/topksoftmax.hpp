#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Topksoftmax, Tensor, Tensor, const Tensor &, const size_t, const int);

/// Top-k softmax: 对最后一维做 softmax 后取 top-k，可选对 top-k 概率再归一化
/// @param values 为 softmax 后的权重
/// @param indices 为所选索引
/// @param x 输入 [N, width]，router logits
/// @param topk 取前 topk 个
/// @param norm 是否对 top-k 概率归一化（1=是，0=否）
void topksoftmax(Tensor values, Tensor indices, const Tensor &x, const size_t topk, const int norm = 0);

} // namespace infinicore::op
