#include "infinicore/ops/fp8_e4m3_dequant.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Fp8E4m3Dequant::schema> &Fp8E4m3Dequant::dispatcher() {
    static common::OpDispatcher<schema> dispatcher;
    return dispatcher;
}

void Fp8E4m3Dequant::execute(Tensor output,
                             const Tensor &input,
                             const Tensor &scales) {
    dispatcher().lookup(input->device().getType())(output, input, scales);
}

Tensor fp8_e4m3_dequant(const Tensor &input,
                        const Tensor &scales,
                        const DataType &output_dtype) {
    auto output = Tensor::empty(input->shape(), output_dtype, input->device());
    fp8_e4m3_dequant_(output, input, scales);
    return output;
}

void fp8_e4m3_dequant_(Tensor output,
                       const Tensor &input,
                       const Tensor &scales) {
    if (!output || !input || !scales) {
        throw std::runtime_error("fp8_e4m3_dequant expects non-empty tensors");
    }
    if (input->device() != output->device()
        || scales->device() != output->device()) {
        throw std::runtime_error("fp8_e4m3_dequant expects tensors on one device");
    }
    if (input->dtype() != DataType::I8 || scales->dtype() != DataType::F32) {
        throw std::runtime_error("fp8_e4m3_dequant expects FP8 bytes and FP32 scales");
    }
    if (output->dtype() != DataType::F16
        && output->dtype() != DataType::BF16
        && output->dtype() != DataType::F32) {
        throw std::runtime_error("fp8_e4m3_dequant expects a floating-point output");
    }
    if (input->shape() != output->shape() || input->ndim() != 2
        || scales->shape() != Shape({input->size(0), 1})) {
        throw std::runtime_error(
            "fp8_e4m3_dequant expects input/output [N,K] and scales [N,1]");
    }
    if (!input->is_contiguous() || !output->is_contiguous()
        || !scales->is_contiguous()) {
        throw std::runtime_error("fp8_e4m3_dequant expects contiguous tensors");
    }
    Fp8E4m3Dequant::execute(output, input, scales);
}

} // namespace infinicore::op
