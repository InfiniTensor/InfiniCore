#include "infinicore/ops/convert_to_f32.hpp"

#include "../../utils.hpp"
#include "infinicore/dtype.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<ConvertToF32::schema> &ConvertToF32::dispatcher() {
    static common::OpDispatcher<ConvertToF32::schema> dispatcher_;
    return dispatcher_;
};

void ConvertToF32::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    if (output->dtype() != DataType::F32) {
        throw std::runtime_error("convert_to_f32: output dtype must be F32");
    }
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No convert_to_f32 implementation found for device type: "
                                 + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

Tensor convert_to_f32(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, DataType::F32, input->device());
    convert_to_f32_(output, input);
    return output;
}

void convert_to_f32_(Tensor output, Tensor input) {
    ConvertToF32::execute(output, input);
}

} // namespace infinicore::op
