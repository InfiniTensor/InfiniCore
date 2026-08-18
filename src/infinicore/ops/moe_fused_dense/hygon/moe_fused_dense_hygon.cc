#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/moe_fused_dense.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <c10/hip/HIPGuard.h>

namespace infinicore::op::moe_fused_dense_impl::hygon {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w13;
    graph::GraphTensor w2;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_ids;
};

void *plan(Tensor output,
           const Tensor &hidden_states,
           const Tensor &w13,
           const Tensor &w2,
           const Tensor &topk_weights,
           const Tensor &topk_ids,
           const Tensor &,
           const Tensor &,
           const Tensor &) {
    return new PlannedMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(hidden_states),
        graph::GraphTensor(w13),
        graph::GraphTensor(w2),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_ids)};
}

void run(void *planned_meta) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    const bool output_need_copy_back = !p->output->is_contiguous();
    Tensor output_work_ic = output_need_copy_back ? p->output->contiguous() : Tensor(p->output);

    auto output = infinicore::adaptor::to_aten_tensor(output_work_ic);
    auto hidden_states = infinicore::adaptor::to_aten_tensor(p->hidden_states);
    auto w13 = infinicore::adaptor::to_aten_tensor(p->w13);
    auto w2 = infinicore::adaptor::to_aten_tensor(p->w2);
    auto topk_weights = infinicore::adaptor::to_aten_tensor(p->topk_weights);
    auto topk_ids = infinicore::adaptor::to_aten_tensor(p->topk_ids);

    const int64_t num_experts = w13.size(0);
    const int64_t intermediate_size = w2.size(2);
    const int64_t topk = topk_ids.size(1);

    auto result = at::zeros_like(output);
    auto topk_ids_i64 = topk_ids.to(at::kLong);

    for (int64_t k = 0; k < topk; ++k) {
        auto ids_k = topk_ids_i64.select(1, k);
        for (int64_t expert = 0; expert < num_experts; ++expert) {
            auto token_indices = at::nonzero(ids_k == expert).flatten();
            if (token_indices.numel() == 0) {
                continue;
            }

            auto hidden = hidden_states.index_select(0, token_indices);
            auto w13_e = w13.select(0, expert);
            auto gate_up = at::matmul(hidden, w13_e.transpose(0, 1));
            auto gate = gate_up.narrow(1, 0, intermediate_size);
            auto up = gate_up.narrow(1, intermediate_size, intermediate_size);
            auto activated = (gate / (1 + at::exp(-gate))) * up;

            auto w2_e = w2.select(0, expert);
            auto expert_out = at::matmul(activated, w2_e.transpose(0, 1));
            auto weights = topk_weights.select(1, k)
                               .index_select(0, token_indices)
                               .to(expert_out.scalar_type())
                               .unsqueeze(1);
            result.index_add_(0, token_indices, expert_out * weights);
        }
    }

    output.copy_(result);
    if (output_need_copy_back) {
        p->output->copy_from(output_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MoeFusedDense::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MoeFusedDense::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MoeFusedDense::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::moe_fused_dense_impl::hygon
#endif // ENABLE_HYGON_API && ENABLE_ATEN
