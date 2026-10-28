#include "../infiniop_impl.hpp"
#include "infinicore/ops/vision_qkv_rope_pad.hpp"
#include "infiniop/ops/vision_qkv_rope_pad.h"
namespace infinicore::op::vision_qkv_rope_pad_impl::infiniop {
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, VisionQKVRoPEPad, 100);
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor q_out;
    graph::GraphTensor k_out;
    graph::GraphTensor v_out;
    graph::GraphTensor qkv;
    graph::GraphTensor cos;
    graph::GraphTensor sin;
    graph::GraphTensor rows;
    graph::GraphTensor cols;
};
void *plan(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &qkv, const Tensor &cos, const Tensor &sin, const Tensor &rows, const Tensor &cols) {
    size_t key = hash_combine(q_out, k_out, v_out, qkv, cos, sin, rows, cols);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, VisionQKVRoPEPad, key, q_out->desc(), k_out->desc(), v_out->desc(), qkv->desc(), cos->desc(), sin->desc(), rows->desc(), cols->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, VisionQKVRoPEPad, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(q_out), graph::GraphTensor(k_out), graph::GraphTensor(v_out), graph::GraphTensor(qkv), graph::GraphTensor(cos), graph::GraphTensor(sin), graph::GraphTensor(rows), graph::GraphTensor(cols)};
}
void run(void *meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(meta);
    INFINICORE_CHECK_ERROR(infiniopVisionQKVRoPEPad(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->q_out->data(), p->k_out->data(), p->v_out->data(), p->qkv->data(), p->cos->data(), p->sin->data(), p->rows->data(), p->cols->data(), context::getStream()));
}
void cleanup(void **meta) {
    delete *reinterpret_cast<PlannedMeta **>(meta);
    *meta = nullptr;
}
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(VisionQKVRoPEPad, &plan, &run, &cleanup);
} // namespace infinicore::op::vision_qkv_rope_pad_impl::infiniop
