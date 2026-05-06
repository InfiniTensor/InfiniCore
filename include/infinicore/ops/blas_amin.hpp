#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class BlasAmin {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor result, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor blas_amin(Tensor x);
void blas_amin_(Tensor result, Tensor x);

} // namespace infinicore::op
