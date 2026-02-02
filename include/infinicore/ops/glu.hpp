#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Glu {
public:
    using schema = void (*)(Tensor, Tensor, int);
    static void execute(Tensor input, Tensor output, int dim);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor glu(Tensor input, int dim);
void glu_(Tensor input, Tensor output, int dim);

} // namespace infinicore::op