#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class VarGlobal {
public:
    using schema = void (*)(Tensor, Tensor, int);
    static void execute(Tensor input, Tensor output, int correction);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor var_global(Tensor input, int correction);
void var_global_(Tensor input, Tensor output, int correction);

class VarReduce {
public:
    using schema = void (*)(Tensor, Tensor, int, int, bool);
    static void execute(Tensor input, Tensor output, int dim, int correction, bool keepdim);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor var_reduce(Tensor input, int dim, int correction, bool keepdim);
void var_reduce_(Tensor input, Tensor output, int dim, int correction, bool keepdim);

} // namespace infinicore::op