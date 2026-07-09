#include "infinicore/ops/deepseek_moe_w8a8i8.hpp"

#include "../../utils.hpp"
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekMoeW8A8I8);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekMoeW8A8I8WithPtrTables);

namespace {

void check_tensors(const std::vector<Tensor> &gate_weights,
                   const std::vector<Tensor> &up_weights,
                   const std::vector<Tensor> &down_weights,
                   const std::vector<Tensor> &gate_weight_scales,
                   const std::vector<Tensor> &up_weight_scales,
                   const std::vector<Tensor> &down_weight_scales,
                   size_t num_experts) {
    if (gate_weights.size() != num_experts || up_weights.size() != num_experts || down_weights.size() != num_experts
        || gate_weight_scales.size() != num_experts || up_weight_scales.size() != num_experts || down_weight_scales.size() != num_experts) {
        throw std::runtime_error("DeepseekMoeW8A8I8: expert tensor vector size mismatch");
    }
}

} // namespace

DeepseekMoeW8A8I8::DeepseekMoeW8A8I8(Tensor out,
                                     const Tensor &hidden,
                                     const Tensor &topk_indices,
                                     const Tensor &topk_weights,
                                     const std::vector<Tensor> &gate_weights,
                                     const std::vector<Tensor> &up_weights,
                                     const std::vector<Tensor> &down_weights,
                                     const std::vector<Tensor> &gate_weight_scales,
                                     const std::vector<Tensor> &up_weight_scales,
                                     const std::vector<Tensor> &down_weight_scales,
                                     size_t intermediate_size,
                                     size_t num_experts) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, hidden, topk_indices, topk_weights);
    check_tensors(gate_weights, up_weights, down_weights, gate_weight_scales, up_weight_scales, down_weight_scales, num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, gate_weights[i], up_weights[i], down_weights[i],
                                              gate_weight_scales[i], up_weight_scales[i], down_weight_scales[i]);
    }
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, hidden, topk_indices, topk_weights,
                                 gate_weights, up_weights, down_weights,
                                 gate_weight_scales, up_weight_scales, down_weight_scales,
                                 intermediate_size, num_experts);
}

void DeepseekMoeW8A8I8::execute(Tensor out,
                                const Tensor &hidden,
                                const Tensor &topk_indices,
                                const Tensor &topk_weights,
                                const std::vector<Tensor> &gate_weights,
                                const std::vector<Tensor> &up_weights,
                                const std::vector<Tensor> &down_weights,
                                const std::vector<Tensor> &gate_weight_scales,
                                const std::vector<Tensor> &up_weight_scales,
                                const std::vector<Tensor> &down_weight_scales,
                                size_t intermediate_size,
                                size_t num_experts) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekMoeW8A8I8,
        out, hidden, topk_indices, topk_weights,
        gate_weights, up_weights, down_weights,
        gate_weight_scales, up_weight_scales, down_weight_scales,
        intermediate_size, num_experts);
}


DeepseekMoeW8A8I8WithPtrTables::DeepseekMoeW8A8I8WithPtrTables(Tensor out,
                                                               const Tensor &hidden,
                                                               const Tensor &topk_indices,
                                                               const Tensor &topk_weights,
                                                               const Tensor &ptr_tables,
                                                               size_t intermediate_size,
                                                               size_t num_experts) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, hidden, topk_indices, topk_weights, ptr_tables);
    const size_t expected_ptr_bytes = num_experts * 6 * sizeof(void *);
    if (ptr_tables->dtype() != DataType::U8 || ptr_tables->numel() < expected_ptr_bytes) {
        throw std::runtime_error("DeepseekMoeW8A8I8WithPtrTables: invalid pointer table tensor");
    }
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, hidden, topk_indices, topk_weights, ptr_tables,
                                 intermediate_size, num_experts);
}

void DeepseekMoeW8A8I8WithPtrTables::execute(Tensor out,
                                             const Tensor &hidden,
                                             const Tensor &topk_indices,
                                             const Tensor &topk_weights,
                                             const Tensor &ptr_tables,
                                             size_t intermediate_size,
                                             size_t num_experts) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekMoeW8A8I8WithPtrTables,
        out, hidden, topk_indices, topk_weights, ptr_tables,
        intermediate_size, num_experts);
}

void deepseek_moe_w8a8i8_(Tensor out,
                          const Tensor &hidden,
                          const Tensor &topk_indices,
                          const Tensor &topk_weights,
                          const std::vector<Tensor> &gate_weights,
                          const std::vector<Tensor> &up_weights,
                          const std::vector<Tensor> &down_weights,
                          const std::vector<Tensor> &gate_weight_scales,
                          const std::vector<Tensor> &up_weight_scales,
                          const std::vector<Tensor> &down_weight_scales,
                          size_t intermediate_size,
                          size_t num_experts) {
    DeepseekMoeW8A8I8::execute(out, hidden, topk_indices, topk_weights,
                               gate_weights, up_weights, down_weights,
                               gate_weight_scales, up_weight_scales, down_weight_scales,
                               intermediate_size, num_experts);
}

Tensor deepseek_moe_w8a8i8(const Tensor &hidden,
                           const Tensor &topk_indices,
                           const Tensor &topk_weights,
                           const std::vector<Tensor> &gate_weights,
                           const std::vector<Tensor> &up_weights,
                           const std::vector<Tensor> &down_weights,
                           const std::vector<Tensor> &gate_weight_scales,
                           const std::vector<Tensor> &up_weight_scales,
                           const std::vector<Tensor> &down_weight_scales,
                           size_t intermediate_size,
                           size_t num_experts) {
    auto out = Tensor::empty(hidden->shape(), hidden->dtype(), hidden->device());
    deepseek_moe_w8a8i8_(out, hidden, topk_indices, topk_weights,
                         gate_weights, up_weights, down_weights,
                         gate_weight_scales, up_weight_scales, down_weight_scales,
                         intermediate_size, num_experts);
    return out;
}


void deepseek_moe_w8a8i8_with_ptr_tables_(Tensor out,
                                          const Tensor &hidden,
                                          const Tensor &topk_indices,
                                          const Tensor &topk_weights,
                                          const Tensor &ptr_tables,
                                          size_t intermediate_size,
                                          size_t num_experts) {
    DeepseekMoeW8A8I8WithPtrTables::execute(out, hidden, topk_indices, topk_weights, ptr_tables,
                                            intermediate_size, num_experts);
}

Tensor deepseek_moe_w8a8i8_with_ptr_tables(const Tensor &hidden,
                                           const Tensor &topk_indices,
                                           const Tensor &topk_weights,
                                           const Tensor &ptr_tables,
                                           size_t intermediate_size,
                                           size_t num_experts) {
    auto out = Tensor::empty(hidden->shape(), hidden->dtype(), hidden->device());
    deepseek_moe_w8a8i8_with_ptr_tables_(out, hidden, topk_indices, topk_weights, ptr_tables,
                                         intermediate_size, num_experts);
    return out;
}

} // namespace infinicore::op
