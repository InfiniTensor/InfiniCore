#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Inner {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor input, Tensor other, Tensor out);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor inner(Tensor input, Tensor other);
void inner_(Tensor input, Tensor other, Tensor out);

}