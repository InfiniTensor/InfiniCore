#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "infinicore/ops/fused_moe_w8a8.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/hygon_aiter_adaptor.hpp"

namespace infinicore::op {
namespace fused_moe_w8a8_impl::hygon {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor selected_experts;
    graph::GraphTensor routing_weights;
    graph::GraphTensor w1;
    graph::GraphTensor w1_scale;
    graph::GraphTensor w2;
    graph::GraphTensor w2_scale;
};

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &selected_experts,
           const Tensor &routing_weights,
           const Tensor &w1,
           const Tensor &w1_scale,
           const Tensor &w2,
           const Tensor &w2_scale) {
    return new PlannedMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        graph::GraphTensor(selected_experts),
        graph::GraphTensor(routing_weights),
        graph::GraphTensor(w1),
        graph::GraphTensor(w1_scale),
        graph::GraphTensor(w2),
        graph::GraphTensor(w2_scale)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    adaptor::set_aten_stream_to_infinicore();
    auto at_output = adaptor::to_aten_tensor(planned->output);
    adaptor::hygon_aiter::fused_moe_w8a8(
        at_output,
        adaptor::to_aten_tensor(planned->input),
        adaptor::to_aten_tensor(planned->w1, at::ScalarType::Float8_e4m3fn),
        adaptor::to_aten_tensor(planned->w2, at::ScalarType::Float8_e4m3fn),
        adaptor::to_aten_tensor(planned->routing_weights),
        adaptor::to_aten_tensor(planned->selected_experts),
        adaptor::to_aten_tensor(planned->w1_scale),
        adaptor::to_aten_tensor(planned->w2_scale));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    FusedMoeW8A8::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    FusedMoeW8A8::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    FusedMoeW8A8::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace fused_moe_w8a8_impl::hygon
} // namespace infinicore::op

#endif
