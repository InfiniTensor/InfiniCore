#pragma once

#include "common/op.hpp"

namespace infinicore::op {
class Zeros {

public:
    using schema = void (*)(Tensor);
    static void execute(Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

void zeros_(Tensor output);
} // namespace infinicore::op
