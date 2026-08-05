#include "infinicore/ops/vocab_parallel_embedding.hpp"
#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"
#include <functional>
#include <memory>
#include <stdexcept>
#if defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
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

void run_vocab_parallel_embedding(
    Tensor out, const Tensor &ids, const Tensor &w, int64_t start, int64_t end) {
#if defined(ENABLE_ATEN)
    adaptor::set_aten_stream_to_infinicore();
    auto o = adaptor::to_aten_tensor(out), i = adaptor::to_aten_tensor(ids), wt = adaptor::to_aten_tensor(w);
    auto mask = i.lt(start).logical_or(i.ge(end));
    auto local = (i - start).clamp(0, end - start - 1);
    auto y = at::embedding(wt, local, -1, false, false);
    y.masked_fill_(mask.unsqueeze(-1), 0);
    o.copy_(y);
#else
    throw std::runtime_error("vocab_parallel_embedding_ requires ATen");
#endif
}

} // namespace

void vocab_parallel_embedding_(Tensor out, const Tensor &ids, const Tensor &w, int64_t start, int64_t end) {
    if (!out || !ids || !w || w->ndim() != 2) {
        throw std::runtime_error("vocab_parallel_embedding_: invalid tensors");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, ids, w);
    if (ids->dtype() != DataType::I32 && ids->dtype() != DataType::I64) {
        throw std::runtime_error("vocab_parallel_embedding_: indices must be int32 or int64");
    }
    if (out->dtype() != w->dtype()) {
        throw std::runtime_error("vocab_parallel_embedding_: output and weight dtypes must match");
    }
    if (start < 0 || end <= start || static_cast<size_t>(end - start) != w->size(0)) {
        throw std::runtime_error("vocab_parallel_embedding_: invalid vocabulary range");
    }
    Shape expected_shape = ids->shape();
    expected_shape.push_back(w->size(1));
    if (out->shape() != expected_shape) {
        throw std::runtime_error("vocab_parallel_embedding_: output shape mismatch");
    }
    record_or_run([out, ids, w, start, end] {
        run_vocab_parallel_embedding(out, ids, w, start, end);
    });
}
} // namespace infinicore::op
