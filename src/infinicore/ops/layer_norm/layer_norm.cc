#include "infinicore/ops/layer_norm.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<LayerNorm::schema> &LayerNorm::dispatcher() {
    static common::OpDispatcher<LayerNorm::schema> dispatcher_;
    return dispatcher_;
};

void LayerNorm::execute(Tensor output,
                        Tensor input_standardization,
                        Tensor input_std_deviation,
                        Tensor input,
                        Tensor weight,
                        Tensor bias,
                        float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input_standardization, input_std_deviation, input, weight, bias);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No LayerNorm implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input_standardization, input_std_deviation, input, weight, bias, epsilon);
}

Tensor layer_norm(Tensor input, Tensor weight, Tensor bias, float epsilon) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());

    if (shape.empty()) {
        throw std::runtime_error("layer_norm: input must have at least one dimension");
    }

    Shape std_shape = shape;
    std_shape.pop_back();
    if (std_shape.empty()) {
        std_shape.push_back(1);
    }

    auto input_standardization = Tensor::empty(shape, input->dtype(), input->device());
    auto input_std_deviation = Tensor::empty(std_shape, input->dtype(), input->device());
    layer_norm_(output, input_standardization, input_std_deviation, input, weight, bias, epsilon);
    return output;
}

void layer_norm_(Tensor output,
                 Tensor input_standardization,
                 Tensor input_std_deviation,
                 Tensor input,
                 Tensor weight,
                 Tensor bias,
                 float epsilon) {
    LayerNorm::execute(output, input_standardization, input_std_deviation, input, weight, bias, epsilon);
}
} // namespace infinicore::op
