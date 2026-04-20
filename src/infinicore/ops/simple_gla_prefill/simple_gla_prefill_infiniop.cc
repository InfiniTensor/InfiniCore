#include "infinicore/ops/simple_gla_prefill.hpp"

#include "../infiniop_impl.hpp"
#include "infinicore/context/context.hpp"

#include <infiniop/ops/simple_gla_prefill.h>

namespace infinicore::op::simple_gla_prefill_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, SimpleGLAPrefill, 64);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor out;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor v;
    graph::GraphTensor g;
    float scale;
};

static void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &g, float scale) {
    size_t key = hash_combine(out, q, k, v, g, static_cast<size_t>(scale * 1000000.0f));

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, SimpleGLAPrefill,
        key, out->desc(), q->desc(), k->desc(), v->desc(), g->desc());

    // Scratch workspace allocation can dominate VRAM growth for long-context prefill.
    // Reuse a single workspace tensor per (workspace_size, device) across layers.
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(
        infiniopGetSimpleGLAPrefillWorkspaceSize(descriptor->desc, &workspace_size));

    thread_local common::OpCache<size_t, Tensor> workspace_caches(8 /*capacity*/);
    auto device__ = context::getDevice();
    auto &cache__ = workspace_caches.getCache(device__);

    Tensor workspace;
    if (auto cached = cache__.get(workspace_size); cached.has_value()) {
        workspace = *cached;
    } else {
        workspace = Tensor::empty({workspace_size}, DataType::U8, device__);
        cache__.put(workspace_size, workspace);
    }

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(g),
        scale,
    };
}

static void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(
        infiniopSimpleGLAPrefill(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->out->data(),
            p->q->data(),
            p->k->data(),
            p->v->data(),
            p->g->data(),
            p->scale,
            context::getStream()));
}

static void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SimpleGLAPrefill, &plan, &run, &cleanup);

} // namespace infinicore::op::simple_gla_prefill_impl::infiniop
