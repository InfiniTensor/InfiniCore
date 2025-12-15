#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class IndexSelect {
public:
    using schema = void (*)(Tensor, Tensor, int, Tensor);
    static void execute(Tensor output, Tensor input, int dim, Tensor index);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor index_select(Tensor input, int dim, Tensor index);
void index_select_(Tensor output, Tensor input, int dim, Tensor index);
} // namespace infinicore::op
