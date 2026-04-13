#include "infinicore/ops/topkrouter.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<TopKRouter::schema> &TopKRouter::dispatcher() {
    static common::OpDispatcher<TopKRouter::schema> dispatcher_;
    return dispatcher_;
};

void TopKRouter::execute(Tensor values_output,
                         Tensor indices_output,
                         Tensor input,
                         Tensor correction_bias,
                         float routed_scaling_factor,
                         size_t topk) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(values_output, indices_output, input, correction_bias);
    infinicore::context::setDevice(input->device());
    auto device_type = input->device().getType();
    auto func = dispatcher().lookup(device_type);
    if (func == nullptr) {
        throw std::runtime_error("No TopKRouter implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }
    func(values_output, indices_output, input, correction_bias, routed_scaling_factor, topk);
}

std::pair<Tensor, Tensor> topkrouter(Tensor input,
                                    Tensor correction_bias,
                                    float routed_scaling_factor,
                                    size_t topk) {
    // values: float32, indices: int32
    auto shape = input->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("topkrouter: input must be 2D [N, width]");
    }
    Tensor values = Tensor::empty({shape[0], topk}, DataType::F32, input->device());
    Tensor indices = Tensor::empty({shape[0], topk}, DataType::I32, input->device());
    topkrouter_(values, indices, input, correction_bias, routed_scaling_factor, topk);
    return {values, indices};
}

void topkrouter_(Tensor values_output,
                 Tensor indices_output,
                 Tensor input,
                 Tensor correction_bias,
                 float routed_scaling_factor,
                 size_t topk) {
    TopKRouter::execute(values_output, indices_output, input, correction_bias, routed_scaling_factor, topk);
}

} // namespace infinicore::op

