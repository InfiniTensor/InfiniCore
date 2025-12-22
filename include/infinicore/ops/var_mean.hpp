#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>
#include <vector>
#include <utility>
namespace infinicore::op {
class Var_Mean {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, std::vector<size_t>, bool, bool); // var_output, mean_output, input, dim, unbiased, keepdim
    static void execute(Tensor var_output, Tensor mean_output, Tensor input, std::vector<size_t> dim, bool unbiased=true, bool keepdim=false);
    static common::OpDispatcher<schema> &dispatcher();
};

// 实现时返回 std::make_pair(var_tensor, mean_tensor) 或使用 C++17 的列表初始化 {var_tensor, mean_tensor}。
// 注意：如果使用 pybind11 绑定到 Python，std::pair 会自动转换为 Python 元组，这与测试文件中的期望一致。
std::pair<Tensor, Tensor> var_mean(Tensor input, std::vector<size_t> dim, bool unbiased=true, bool keepdim = false);
void var_mean_(Tensor var_output, Tensor mean_output, Tensor input, std::vector<size_t> dim, bool unbiased=true, bool keepdim = false);

} // namespace infinicore::op
