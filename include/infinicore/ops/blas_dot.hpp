#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class BlasDot {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor result, Tensor x, Tensor y);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor blas_dot(Tensor x, Tensor y);
void blas_dot_(Tensor result, Tensor x, Tensor y);

} // namespace infinicore::op
