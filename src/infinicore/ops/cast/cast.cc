#include "infinicore/ops/cast.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

#include <functional>
#include <memory>
#include <stdexcept>

#if defined(ENABLE_ATEN)
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

void run_cast(Tensor out, const Tensor &in) {
#if defined(ENABLE_ATEN)
    adaptor::set_aten_stream_to_infinicore();
    auto output_at = adaptor::to_aten_tensor(out);
    const auto input_at = adaptor::to_aten_tensor(in);
    output_at.copy_(input_at);
#else
    throw std::runtime_error("cast_ requires ATen");
#endif
}

} // namespace

void cast_(Tensor out, const Tensor &in) {
    if (!out || !in || out->shape() != in->shape()) {
        throw std::runtime_error("cast_ expects equal non-empty shapes");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, in);
    record_or_run([out, in] { run_cast(out, in); });
}

} // namespace infinicore::op
