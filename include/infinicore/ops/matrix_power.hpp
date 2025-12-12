#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class MatrixPower {
public:
    using schema = void (*)(Tensor, Tensor, int);
    static void execute(Tensor output, Tensor input, int n);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor matrix_power(Tensor input, int n);
void matrix_power_(Tensor output, Tensor input, int n);

} // namespace infinicore::op

