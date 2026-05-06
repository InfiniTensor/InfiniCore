#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class BlasAmax {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor result, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor blas_amax(Tensor x);
void blas_amax_(Tensor result, Tensor x);

} // namespace infinicore::op
