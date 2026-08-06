#include "infinicore/ops/scaled_mm_w4a16_awq.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

#include "../../utils.hpp"
#include "../vendor_ops/vendor_ops_dispatch.hpp"

#include <functional>
#include <stdexcept>

namespace infinicore::op {
namespace {

class ScaledMmW4A16AwqGraphOperator final : public graph::GraphOperator {
public:
    explicit ScaledMmW4A16AwqGraphOperator(std::function<void()> runner)
        : runner_(std::move(runner)) {}

    void run() const override {
        runner_();
    }

private:
    std::function<void()> runner_;
};

} // namespace

Tensor scaled_mm_w4a16_awq(const Tensor &input, const Tensor &qweight,
                           const Tensor &qzeros, const Tensor &scales,
                           std::optional<Tensor> bias) {
    if (input->ndim() != 2 || qweight->ndim() != 2) {
        throw std::runtime_error("scaled_mm_w4a16_awq expects 2D input and qweight");
    }
    auto out = Tensor::empty({input->size(0), qweight->size(1) * 2},
                             input->dtype(), input->device());
    scaled_mm_w4a16_awq_(out, input, qweight, qzeros, scales, bias);
    return out;
}

void scaled_mm_w4a16_awq_(Tensor out, const Tensor &input,
                          const Tensor &qweight, const Tensor &qzeros,
                          const Tensor &scales, std::optional<Tensor> bias) {
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, qweight, qzeros, scales, *bias);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, qweight, qzeros, scales);
    }
    if (out->ndim() != 2 || input->ndim() != 2 || qweight->ndim() != 2
        || qzeros->ndim() != 2 || scales->ndim() != 2) {
        throw std::runtime_error("scaled_mm_w4a16_awq expects 2D tensors");
    }
    if (input->dtype() != DataType::BF16 || out->dtype() != DataType::BF16) {
        throw std::runtime_error("scaled_mm_w4a16_awq currently expects bfloat16 input and out");
    }
    if (qweight->dtype() != DataType::I8 || qzeros->dtype() != DataType::I8
        || scales->dtype() != DataType::BF16) {
        throw std::runtime_error("scaled_mm_w4a16_awq expects int8 qweight/qzeros and bfloat16 scales");
    }
    const size_t m = input->size(0);
    const size_t k = input->size(1);
    const size_t n = qweight->size(1) * 2;
    if ((k % 256) != 0 || qweight->size(0) != k || (n % 2) != 0) {
        throw std::runtime_error("scaled_mm_w4a16_awq invalid K or qweight shape");
    }
    if (out->size(0) != m || out->size(1) != n) {
        throw std::runtime_error("scaled_mm_w4a16_awq out shape mismatch");
    }
    if (qzeros->size(0) != k / 64 || qzeros->size(1) != n / 2) {
        throw std::runtime_error("scaled_mm_w4a16_awq expects qzeros [K/64,N/2]");
    }
    if (scales->size(0) != k / 64 || scales->size(1) != n) {
        throw std::runtime_error("scaled_mm_w4a16_awq expects scales [K/64,N]");
    }
    if (bias && ((*bias)->ndim() != 1 || (*bias)->size(0) != n || (*bias)->dtype() != out->dtype())) {
        throw std::runtime_error("scaled_mm_w4a16_awq invalid bias");
    }
    if (!out->is_contiguous() || !input->is_contiguous()
        || !qweight->is_contiguous() || !qzeros->is_contiguous()
        || !scales->is_contiguous() || (bias && !(*bias)->is_contiguous())) {
        throw std::runtime_error("scaled_mm_w4a16_awq expects contiguous tensors");
    }
    if (context::isGraphRecording()) {
        context::addGraphOperator(
            std::make_shared<ScaledMmW4A16AwqGraphOperator>(
                [out, input, qweight, qzeros, scales, bias] {
                    scaled_mm_w4a16_awq_(
                        out, input, qweight, qzeros, scales, bias);
                }));
        return;
    }
    auto kernel = vendor_ops::lookup(
        vendor_ops::scaled_mm_w4a16_awq_dispatcher(), out->device().getType(),
        "scaled_mm_w4a16_awq");
    kernel(out, input, qweight, qzeros, scales, bias);
}

} // namespace infinicore::op
