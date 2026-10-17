#include "infinicore/ops/linear_w8a16.hpp"
#include "../../utils.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#if defined(ENABLE_ILUVATAR_FLASH_ATTN)
#include <ATen/ATen.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace ixformer {
namespace infer {
void w8a16_gemm(at::Tensor &,
                at::Tensor &,
                at::Tensor &,
                at::Tensor &,
                long,
                std::string,
                long,
                const std::optional<at::Tensor> &,
                long);
} // namespace infer
} // namespace ixformer
#endif

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LinearW8A16);

LinearW8A16::LinearW8A16(Tensor out,
                         const Tensor &input,
                         const Tensor &weight,
                         const Tensor &scale,
                         std::optional<Tensor> bias) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight, scale);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, input, weight, scale, bias);
}

void LinearW8A16::execute(Tensor out,
                          const Tensor &input,
                          const Tensor &weight,
                          const Tensor &scale,
                          std::optional<Tensor> bias) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        LinearW8A16, out, input, weight, scale, bias);
}

namespace {
struct PlannedMeta {
    graph::GraphTensor out, input, weight, scale;
    std::optional<graph::GraphTensor> bias;
};

void validate(Tensor out,
              const Tensor &input,
              const Tensor &weight,
              const Tensor &scale,
              const std::optional<Tensor> &bias) {
    if (input->ndim() != 2 || out->ndim() != 2 ||
        input->size(1) != weight->size(1) || out->size(1) != weight->size(0) ||
        out->size(0) != input->size(0) || !out->is_contiguous() ||
        !input->is_contiguous() || !weight->is_contiguous() ||
        !scale->is_contiguous()) {
        throw std::runtime_error(
            "LinearW8A16 expects contiguous [M,K] input/out and [N,K] weight");
    }
    if (weight->dtype() != DataType::I8 || scale->ndim() != 2 ||
        scale->size(0) != 1 || scale->size(1) != weight->size(0) ||
        scale->dtype() != input->dtype() ||
        (input->dtype() != DataType::F16 &&
         input->dtype() != DataType::BF16) ||
        out->dtype() != input->dtype()) {
        throw std::runtime_error("LinearW8A16 has invalid weight or scale");
    }
    if (bias.has_value() &&
        (bias.value()->ndim() != 1 ||
         bias.value()->size(0) != weight->size(0) ||
         bias.value()->dtype() != input->dtype())) {
        throw std::runtime_error("LinearW8A16 has invalid bias");
    }
}
} // namespace

namespace linear_w8a16_impl {
void *plan(Tensor out,
           const Tensor &input,
           const Tensor &weight,
           const Tensor &scale,
           std::optional<Tensor> bias) {
    validate(out, input, weight, scale, bias);
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(weight),
        graph::GraphTensor(scale),
        bias ? std::optional<graph::GraphTensor>(graph::GraphTensor(*bias))
             : std::nullopt};
}

void run(void *planned_meta_ptr) {
#if defined(ENABLE_ILUVATAR_FLASH_ATTN)
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta_ptr);
    auto output = adaptor::to_aten_tensor(p->out);
    auto input = adaptor::to_aten_tensor(p->input);
    auto weight = adaptor::to_aten_tensor(p->weight);
    auto scale = adaptor::to_aten_tensor(p->scale);
    auto bias = p->bias
                    ? std::optional<at::Tensor>(
                          adaptor::to_aten_tensor(*p->bias))
                    : std::nullopt;
    adaptor::set_aten_stream_to_infinicore();
    ixformer::infer::w8a16_gemm(output,
                                input,
                                weight,
                                scale,
                                -1,
                                "TN",
                                0,
                                bias,
                                -1);
#else
    (void)planned_meta_ptr;
    throw std::runtime_error("LinearW8A16 is not enabled in this build");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    LinearW8A16::plan_dispatcher().registerAll(&plan, false);
    LinearW8A16::run_dispatcher().registerAll(&run, false);
    LinearW8A16::cleanup_dispatcher().registerAll(&cleanup, false);
    return true;
}();
} // namespace linear_w8a16_impl

Tensor linear_w8a16(const Tensor &input,
                    const Tensor &weight,
                    const Tensor &scale,
                    std::optional<Tensor> bias) {
    auto output_shape = input->shape();
    output_shape.back() = weight->size(0);
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());
    linear_w8a16_(out, input, weight, scale, bias);
    return out;
}

void linear_w8a16_(Tensor out,
                   const Tensor &input,
                   const Tensor &weight,
                   const Tensor &scale,
                   std::optional<Tensor> bias) {
    const size_t in_features = input->size(input->ndim() - 1);
    const size_t out_features = weight->size(0);
    const size_t rows = input->numel() / in_features;
    LinearW8A16::execute(out->view({rows, out_features}),
                         input->view({rows, in_features}),
                         weight,
                         scale,
                         bias);
}

} // namespace infinicore::op
