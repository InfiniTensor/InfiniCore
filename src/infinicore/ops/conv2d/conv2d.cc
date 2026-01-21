#include "infinicore/ops/conv2d.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Conv2d::schema> &Conv2d::dispatcher() {
    static common::OpDispatcher<Conv2d::schema> dispatcher_;
    return dispatcher_;
};

void Conv2d::execute(Tensor output,
                     Tensor input,
                     Tensor weight,
                     Tensor bias,
                     const size_t *pads,
                     const size_t *strides,
                     const size_t *dilations,
                     size_t n) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight, bias);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Conv2d implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, weight, bias, pads, strides, dilations, n);
}

Tensor conv2d(Tensor input,
              Tensor weight,
              Tensor bias,
              const std::vector<size_t> &pads,
              const std::vector<size_t> &strides,
              const std::vector<size_t> &dilations) {
    // Output shape should be pre-computed by caller; allocate a conservative placeholder.
    // This helper is rarely used in performance-critical paths.
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    conv2d_(output, input, weight, bias, pads, strides, dilations);
    return output;
}

void conv2d_(Tensor output,
             Tensor input,
             Tensor weight,
             Tensor bias,
             const std::vector<size_t> &pads,
             const std::vector<size_t> &strides,
             const std::vector<size_t> &dilations) {
    if (pads.size() != strides.size() || pads.size() != dilations.size()) {
        throw std::runtime_error("conv2d_: pads/strides/dilations must have the same size");
    }
    Conv2d::execute(output,
                    input,
                    weight,
                    bias,
                    pads.data(),
                    strides.data(),
                    dilations.data(),
                    pads.size());
}
} // namespace infinicore::op
