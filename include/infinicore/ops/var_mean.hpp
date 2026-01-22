#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <tuple>

namespace infinicore::op {

class VarMeanGlobal {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, int);
    static void execute(Tensor input, Tensor out_var, Tensor out_mean, int correction);
    static common::OpDispatcher<schema> &dispatcher();
};

std::tuple<Tensor, Tensor> var_mean_global(Tensor input, int correction);
void var_mean_global_(Tensor input, Tensor out_var, Tensor out_mean, int correction);

class VarMeanReduce {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, int, int, bool);
    static void execute(Tensor input, Tensor out_var, Tensor out_mean, int dim, int correction, bool keepdim);
    static common::OpDispatcher<schema> &dispatcher();
};

std::tuple<Tensor, Tensor> var_mean_reduce(Tensor input, int dim, int correction, bool keepdim);
void var_mean_reduce_(Tensor input, Tensor out_var, Tensor out_mean, int dim, int correction, bool keepdim);

} // namespace infinicore::op