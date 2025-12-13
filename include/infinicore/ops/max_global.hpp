#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class MaxGlobal {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor input, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor max_global(Tensor input);
void max_global_(Tensor input, Tensor output);

} // namespace infinicore::op
