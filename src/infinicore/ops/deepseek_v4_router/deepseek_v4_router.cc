#include "infinicore/ops/deepseek_v4_router.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4TopkRouter);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4HashRouter);

DeepseekV4TopkRouter::DeepseekV4TopkRouter(Tensor topk_weights,
                                           Tensor topk_indices,
                                           const Tensor &logits,
                                           const Tensor &bias,
                                           bool renormalize) {
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_indices, logits, bias);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_indices, logits);
    }
    INFINICORE_GRAPH_OP_DISPATCH(
        topk_weights->device().getType(),
        topk_weights,
        topk_indices,
        logits,
        bias,
        renormalize);
}

void DeepseekV4TopkRouter::execute(Tensor topk_weights,
                                   Tensor topk_indices,
                                   const Tensor &logits,
                                   const Tensor &bias,
                                   bool renormalize) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4TopkRouter,
        topk_weights,
        topk_indices,
        logits,
        bias,
        renormalize);
}

std::tuple<Tensor, Tensor> deepseek_v4_topk_router(
    const Tensor &logits,
    size_t topk,
    bool renormalize,
    const Tensor &bias) {
    auto shape = logits->shape();
    INFINICORE_ASSERT(shape.size() == 2);
    auto topk_weights = Tensor::empty({shape[0], topk}, DataType::F32, logits->device());
    auto topk_indices = Tensor::empty({shape[0], topk}, DataType::I32, logits->device());
    deepseek_v4_topk_router_(topk_weights, topk_indices, logits, bias, renormalize);
    return {topk_weights, topk_indices};
}

void deepseek_v4_topk_router_(Tensor topk_weights,
                              Tensor topk_indices,
                              const Tensor &logits,
                              const Tensor &bias,
                              bool renormalize) {
    DeepseekV4TopkRouter::execute(topk_weights, topk_indices, logits, bias, renormalize);
}

DeepseekV4HashRouter::DeepseekV4HashRouter(Tensor topk_weights,
                                           Tensor topk_indices,
                                           const Tensor &logits,
                                           const Tensor &input_ids,
                                           const Tensor &tid2eid,
                                           bool renormalize) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_indices, logits, input_ids, tid2eid);
    INFINICORE_GRAPH_OP_DISPATCH(
        topk_weights->device().getType(),
        topk_weights,
        topk_indices,
        logits,
        input_ids,
        tid2eid,
        renormalize);
}

void DeepseekV4HashRouter::execute(Tensor topk_weights,
                                   Tensor topk_indices,
                                   const Tensor &logits,
                                   const Tensor &input_ids,
                                   const Tensor &tid2eid,
                                   bool renormalize) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4HashRouter,
        topk_weights,
        topk_indices,
        logits,
        input_ids,
        tid2eid,
        renormalize);
}

std::tuple<Tensor, Tensor> deepseek_v4_hash_router(
    const Tensor &logits,
    const Tensor &input_ids,
    const Tensor &tid2eid,
    bool renormalize) {
    auto logits_shape = logits->shape();
    auto table_shape = tid2eid->shape();
    INFINICORE_ASSERT(logits_shape.size() == 2);
    INFINICORE_ASSERT(table_shape.size() == 2);
    auto topk = table_shape[1];
    auto topk_weights = Tensor::empty({logits_shape[0], topk}, DataType::F32, logits->device());
    auto topk_indices = Tensor::empty({logits_shape[0], topk}, DataType::I32, logits->device());
    deepseek_v4_hash_router_(topk_weights, topk_indices, logits, input_ids, tid2eid, renormalize);
    return {topk_weights, topk_indices};
}

void deepseek_v4_hash_router_(Tensor topk_weights,
                              Tensor topk_indices,
                              const Tensor &logits,
                              const Tensor &input_ids,
                              const Tensor &tid2eid,
                              bool renormalize) {
    DeepseekV4HashRouter::execute(topk_weights, topk_indices, logits, input_ids, tid2eid, renormalize);
}

} // namespace infinicore::op
