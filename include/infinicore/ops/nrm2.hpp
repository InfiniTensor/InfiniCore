#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Nrm2 {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor result, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor nrm2(Tensor x);
void nrm2_(Tensor result, Tensor x);

} // namespace infinicore::op
