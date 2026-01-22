#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class SumGlobal {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor input, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor sum_global(Tensor input);
void sum_global_(Tensor input, Tensor output);

class SumReduce {
public:
    using schema = void (*)(Tensor, Tensor, int, bool);
    static void execute(Tensor input, Tensor output, int dim, bool keepdim);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor sum_reduce(Tensor input, int dim, bool keepdim);
void sum_reduce_(Tensor input, Tensor output, int dim, bool keepdim);

} // namespace infinicore::op