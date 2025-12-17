#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Histc {
public:
    using schema = void (*)(Tensor, Tensor, size_t, double, double);
    static void execute(Tensor input, Tensor output, size_t bins, double min, double max);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor histc(Tensor input, size_t bins, double min, double max);
void histc_(Tensor input, Tensor output, size_t bins, double min, double max);
} // namespace infinicore::op
