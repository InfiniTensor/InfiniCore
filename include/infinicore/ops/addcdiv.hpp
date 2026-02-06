#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Addcdiv {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float);
    static void execute(Tensor input, Tensor t1, Tensor t2, Tensor output, float value);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor addcdiv(Tensor input, Tensor t1, Tensor t2, float value);
void addcdiv_(Tensor input, Tensor t1, Tensor t2, Tensor output, float value);

} // namespace infinicore::op