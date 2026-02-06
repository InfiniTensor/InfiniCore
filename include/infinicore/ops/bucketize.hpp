#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Bucketize {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, bool);
    static void execute(Tensor input, Tensor boundaries, Tensor output, bool right);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor bucketize(Tensor input, Tensor boundaries, bool right = false);
void bucketize_(Tensor input, Tensor boundaries, Tensor output, bool right = false);

} // namespace infinicore::op