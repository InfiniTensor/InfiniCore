#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>
#include <vector>
// 如果是类似torch.sum的话，输入输出都应该是Tensor！！！
// using SumSchema = void (*)(T, Tensor, std::optional<int32_t>, bool);
namespace infinicore::op {
class Sum {
public:
    using schema = void (*)(Tensor, Tensor, std::vector<size_t>, bool);
    static void execute(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim=false);
    static common::OpDispatcher<schema> &dispatcher();
};


Tensor sum(Tensor input, std::vector<size_t> dim, bool keepdim = false);
void sum_(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim = false);


// legacy version (for compatibility)
// T sum(Tensor input); // 要template
// void sum_(Tensor output, Tensor input);
} // namespace infinicore::op
