#include "../infiniop_impl.hpp"
#include "infinicore/ops/qk_rms_norm_mrope.hpp"
#include "infiniop/ops/qk_rms_norm_mrope.h"
namespace infinicore::op::qk_rms_norm_mrope_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, QKRMSNormMRoPE, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor q_out;
    graph::GraphTensor k_out;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor q_weight;
    graph::GraphTensor k_weight;
    graph::GraphTensor cos;
    graph::GraphTensor sin;
    graph::GraphTensor positions;
};
void *plan(Tensor q_out,
           Tensor k_out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &q_weight,
           const Tensor &k_weight,
           const Tensor &cos,
           const Tensor &sin,
           const Tensor &positions, float epsilon, int section_t, int section_h, int section_w, bool interleaved) {
    size_t key = hash_combine(q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions, epsilon, section_t, section_h, section_w, interleaved);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, QKRMSNormMRoPE, key, q_out->desc(), k_out->desc(), q->desc(), k->desc(), q_weight->desc(), k_weight->desc(), cos->desc(), sin->desc(), positions->desc(), epsilon, section_t, section_h, section_w, interleaved);
    INFINIOP_WORKSPACE_TENSOR(workspace, QKRMSNormMRoPE, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(q_out), graph::GraphTensor(k_out), graph::GraphTensor(q), graph::GraphTensor(k), graph::GraphTensor(q_weight), graph::GraphTensor(k_weight), graph::GraphTensor(cos), graph::GraphTensor(sin), graph::GraphTensor(positions)};
}
void run(void *meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(meta);
    INFINICORE_CHECK_ERROR(infiniopQKRMSNormMRoPE(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->q_out->data(), p->k_out->data(), p->q->data(), p->k->data(), p->q_weight->data(), p->k_weight->data(), p->cos->data(), p->sin->data(), p->positions->data(), context::getStream()));
}
void cleanup(void **meta) {
    delete *reinterpret_cast<PlannedMeta **>(meta);
    *meta = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(QKRMSNormMRoPE, &plan, &run, &cleanup);
} // namespace infinicore::op::qk_rms_norm_mrope_impl::infiniop
