#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class AllGlobal {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor input, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor all_global(Tensor input);
void all_global_(Tensor input, Tensor output);

class AllReduce {
public:
    using schema = void (*)(Tensor, Tensor, int, bool);
    static void execute(Tensor input, Tensor output, int dim, bool keepdim);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor all_reduce(Tensor input, int dim, bool keepdim);
void all_reduce_(Tensor input, Tensor output, int dim, bool keepdim);

} // namespace infinicore::op