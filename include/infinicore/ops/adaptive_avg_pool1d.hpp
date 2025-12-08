#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class AdaptiveAvgPool1d {
public:
    // Schema: execute(Output, Input)
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

// 函数接口
// output_size: 目标输出长度 (L_out)
Tensor adaptive_avg_pool1d(Tensor input, int64_t output_size);

} // namespace infinicore::op