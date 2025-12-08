#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Addbmm {
public:
    // Schema: execute(Output, Input, Batch1, Batch2, beta, alpha)
    // 【关键修复】这里必须是 float, float，顺序对应下面的 execute
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float, float);
    
    // 【关键修复】参数顺序改为: beta 在前, alpha 在后
    static void execute(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha);
    
    static common::OpDispatcher<schema> &dispatcher();
};

// 1. Out-of-place 接口
// 【关键修复】参数顺序改为: beta 在前, alpha 在后
Tensor addbmm(Tensor input, Tensor batch1, Tensor batch2, float beta = 1.0f, float alpha = 1.0f);

// 2. In-place 接口
// 【关键修复】参数顺序改为: beta 在前, alpha 在后
void addbmm_(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha);

} // namespace infinicore::op