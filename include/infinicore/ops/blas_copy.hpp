#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class BlasCopy {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor x, Tensor y);
    static common::OpDispatcher<schema> &dispatcher();
};

void blas_copy_(Tensor x, Tensor y);

} // namespace infinicore::op
