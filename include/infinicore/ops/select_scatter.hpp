#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class SelectScatter {
public:
    using schema = void (*)(Tensor, Tensor, int64_t, int64_t, Tensor);
    static void execute(Tensor input, Tensor src, int64_t dim, int64_t index, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor select_scatter(Tensor input, Tensor src, int64_t dim, int64_t index);
void select_scatter_(Tensor input, Tensor src, int64_t dim, int64_t index, Tensor output);

} // namespace infinicore::op