#pragma once

#include "common/op.hpp"

namespace infinicore::op {

class Embedding {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor out, Tensor input, Tensor weight);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor embedding(Tensor input, Tensor weight);
void embedding_(Tensor out, Tensor input, Tensor weight);
} // namespace infinicore::op
