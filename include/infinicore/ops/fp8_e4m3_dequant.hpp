#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Fp8E4m3Dequant {
public:
    using schema = void (*)(Tensor, const Tensor &, const Tensor &);

    static void execute(Tensor output,
                        const Tensor &input,
                        const Tensor &scales);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor fp8_e4m3_dequant(const Tensor &input,
                        const Tensor &scales,
                        const DataType &output_dtype);
void fp8_e4m3_dequant_(Tensor output,
                       const Tensor &input,
                       const Tensor &scales);

} // namespace infinicore::op
