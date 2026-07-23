#include "infinicore/ops/bmm_strided.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

#include <functional>
#include <memory>
#include <stdexcept>

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#endif

namespace infinicore::op {
namespace {

class DeferredGraphOperator final : public graph::GraphOperator {
public:
    explicit DeferredGraphOperator(std::function<void()> runner)
        : runner_(std::move(runner)) {}

    void run() const override { runner_(); }

private:
    std::function<void()> runner_;
};

void record_or_run(std::function<void()> runner) {
    auto op = std::make_shared<DeferredGraphOperator>(std::move(runner));
    if (context::isGraphRecording()) {
        context::addGraphOperator(op);
    } else {
        op->run();
    }
}

void validate_bmm_strided(const Tensor &output, const Tensor &a, const Tensor &b) {
    if (!output || !a || !b) {
        throw std::runtime_error("bmm_strided expects non-empty output, a, and b tensors");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, a, b);
    if (output->dtype() != a->dtype() || output->dtype() != b->dtype()) {
        throw std::runtime_error("bmm_strided expects output, a, and b to have the same dtype");
    }
    if (output->dtype() != DataType::F16
        && output->dtype() != DataType::BF16
        && output->dtype() != DataType::F32) {
        throw std::runtime_error("bmm_strided expects float16, bfloat16, or float32 tensors");
    }
    if (output->ndim() != 3 || a->ndim() != 3 || b->ndim() != 3) {
        throw std::runtime_error("bmm_strided expects three-dimensional tensors");
    }
    if (a->size(0) != b->size(0)
        || a->size(0) != output->size(0)
        || a->size(2) != b->size(1)
        || a->size(1) != output->size(1)
        || b->size(2) != output->size(2)) {
        throw std::runtime_error("bmm_strided tensor shapes are incompatible");
    }
}

void run_bmm_strided(Tensor output, const Tensor &a, const Tensor &b) {
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (output->device().getType() == Device::Type::ILUVATAR) {
        adaptor::set_aten_stream_to_infinicore();
        auto output_at = adaptor::to_aten_tensor(output);
        const auto a_at = adaptor::to_aten_tensor(a);
        const auto b_at = adaptor::to_aten_tensor(b);
        at::bmm_out(output_at, a_at, b_at);
        return;
    }
#endif

    throw std::runtime_error(
        "bmm_strided currently supports only Iluvatar builds with ATen");
}

} // namespace

void bmm_strided_(Tensor output, const Tensor &a, const Tensor &b) {
    validate_bmm_strided(output, a, b);
    record_or_run([output, a, b] {
        run_bmm_strided(output, a, b);
    });
}

} // namespace infinicore::op
