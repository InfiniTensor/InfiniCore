#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Gcd {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor input, Tensor other, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor gcd(Tensor input, Tensor other);
void gcd_(Tensor input, Tensor other, Tensor output);

} // namespace infinicore::op