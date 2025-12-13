#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <tuple>

namespace infinicore::op {

class MaxReduce {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, int, bool);
    static void execute(Tensor input, Tensor output, Tensor indices, int dim, bool keepdim);
    static common::OpDispatcher<schema> &dispatcher();
};

std::tuple<Tensor, Tensor> max_reduce(Tensor input, int dim, bool keepdim);
void max_reduce_(Tensor input, Tensor output, Tensor indices, int dim, bool keepdim);

} // namespace infinicore::op
