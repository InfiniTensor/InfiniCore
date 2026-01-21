#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// 这个宏会自动定义 SiluAndMul 类，并包含：
// execute, dispatcher, plan_dispatcher, run_dispatcher, cleanup_dispatcher
// 以及对应的 schema 类型定义
INFINICORE_GRAPH_OP_CLASS(SiluAndMul, Tensor, Tensor);

// 全局辅助函数
Tensor silu_and_mul(Tensor x);
void silu_and_mul_(Tensor out, Tensor x);

} // namespace infinicore::op
