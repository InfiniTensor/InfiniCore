#include "infinicore/ops/float_power.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/tensor.hpp"

namespace infinicore::op {
namespace {

class RecordedScalarPower final : public graph::GraphOperator {
public:
    RecordedScalarPower(Tensor output, Tensor input, double exponent)
        : output_(output), input_(input), exponent_(exponent) {}

    void run() const override {
        FloatPower::dispatcher_scalar().lookup(input_->device().getType())(output_, input_, exponent_);
    }

private:
    graph::GraphTensor output_, input_;
    double exponent_;
};

} // namespace

// =======================================================================
// 1. Dispatcher 单例
// =======================================================================

common::OpDispatcher<FloatPower::schema_scalar> &FloatPower::dispatcher_scalar() {
    static common::OpDispatcher<FloatPower::schema_scalar> dispatcher_;
    return dispatcher_;
}

common::OpDispatcher<FloatPower::schema_tensor> &FloatPower::dispatcher_tensor() {
    static common::OpDispatcher<FloatPower::schema_tensor> dispatcher_;
    return dispatcher_;
}

// =======================================================================
// 2. Execute (执行入口)
// =======================================================================

void FloatPower::execute(Tensor output, Tensor input, double exponent) {
    if (context::isGraphRecording()) {
        context::addGraphOperator(std::make_shared<RecordedScalarPower>(output, input, exponent));
        return;
    }
    dispatcher_scalar()
        .lookup(context::getDevice().getType())(output, input, exponent);
}

void FloatPower::execute(Tensor output, Tensor input, Tensor exponent) {
    dispatcher_tensor()
        .lookup(context::getDevice().getType())(output, input, exponent);
}

// =======================================================================
// 3. Functional interface (out-of-place) -> 强制提升为 F64
// =======================================================================

Tensor float_power(Tensor input, double exponent) {
    auto output = Tensor::empty(
        input->shape(),
        infinicore::DataType::F64,
        input->device());

    float_power_(output, input, exponent);
    return output;
}

Tensor float_power(Tensor input, Tensor exponent) {
    Shape output_shape = input->shape();
    auto output = Tensor::empty(
        output_shape,
        infinicore::DataType::F64,
        input->device());

    float_power_(output, input, exponent);
    return output;
}

// =======================================================================
// 4. Explicit / in-place
// =======================================================================

void float_power_(Tensor output, Tensor input, double exponent) {
    FloatPower::execute(output, input, exponent);
}

void float_power_(Tensor output, Tensor input, Tensor exponent) {
    FloatPower::execute(output, input, exponent);
}

} // namespace infinicore::op
