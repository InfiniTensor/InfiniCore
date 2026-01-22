#pragma once
#include "../device.hpp"
#include "common/op.hpp"
#include <tuple>

namespace infinicore::op {

class TopK {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, int, int, bool, bool);
    static void execute(Tensor input, Tensor values, Tensor indices, int k, int dim, bool largest, bool sorted);
    static common::OpDispatcher<schema> &dispatcher();
};

std::tuple<Tensor, Tensor> topk(Tensor input, int k, int dim, bool largest, bool sorted);
void topk_(Tensor input, Tensor values, Tensor indices, int k, int dim, bool largest, bool sorted);

} // namespace infinicore::op