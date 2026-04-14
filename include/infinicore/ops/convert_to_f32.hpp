#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class ConvertToF32 {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor convert_to_f32(Tensor input);
void convert_to_f32_(Tensor output, Tensor input);

} // namespace infinicore::op
