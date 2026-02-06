#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Atan2 {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor input, Tensor other, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor atan2(Tensor input, Tensor other);
void atan2_(Tensor input, Tensor other, Tensor output);

} // namespace infinicore::op