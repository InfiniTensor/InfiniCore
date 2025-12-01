#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Diagflat {
public:
    using schema = void (*)(Tensor, Tensor, int64_t);
    static void execute(Tensor output, Tensor input, int64_t offset);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor diagflat(Tensor input, int64_t offset = 0);
void diagflat_(Tensor output, Tensor input, int64_t offset = 0);

} // namespace infinicore::op


