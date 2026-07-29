#include "infinicore/ops/ernie45_rope.hpp"

#include "../infiniop_impl.hpp"
#include "infiniop/ops/ernie45_rope.h"

namespace infinicore::op::ernie45_rope_impl::infiniop {

namespace mrope {
INFINIOP_CACHABLE_DESCRIPTOR(MropeDescriptor, Ernie45Mrope, 100);
} // namespace mrope

namespace vision {
INFINIOP_CACHABLE_DESCRIPTOR(VisionDescriptor, Ernie45VisionRope, 100);
} // namespace vision

namespace mrope_op {

struct MropePlannedMeta {
    std::shared_ptr<mrope::MropeDescriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor positions;
};

void *plan_mrope(Tensor q,
                 Tensor k,
                 const Tensor &positions,
                 double rope_theta,
                 size_t section_h,
                 size_t section_w,
                 size_t section_t) {
    size_t key = hash_combine(q, k, positions, rope_theta, section_h, section_w, section_t);
    std::shared_ptr<mrope::MropeDescriptor> descriptor;
    {
        auto device__ = context::getDevice();
        auto &cache__ = mrope::caches.getCache(device__);
        descriptor = cache__.get(key).value_or(nullptr);
        if (!descriptor) {
            descriptor = std::make_shared<mrope::MropeDescriptor>(nullptr);
            INFINICORE_CHECK_ERROR(infiniopCreateErnie45MropeDescriptor(
                context::getInfiniopHandle(device__),
                &descriptor->desc,
                q->desc(), k->desc(), positions->desc(), rope_theta, section_h, section_w, section_t));
            cache__.put(key, descriptor);
        }
    }
    INFINIOP_WORKSPACE_TENSOR(workspace, Ernie45Mrope, descriptor);
    return new MropePlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(positions)};
}

void run_mrope(void *planned_meta) {
    auto *p = reinterpret_cast<MropePlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(
        infiniopErnie45Mrope(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->q->data(),
            p->k->data(),
            p->positions->data(),
            context::getStream()));
}

void cleanup_mrope(void **planned_meta_ptr) {
    delete *reinterpret_cast<MropePlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Ernie45MRoPE, &plan_mrope, &run_mrope, &cleanup_mrope);

} // namespace mrope_op

namespace vision_op {

struct VisionPlannedMeta {
    std::shared_ptr<vision::VisionDescriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor positions;
};

void *plan_vision(Tensor q,
                  Tensor k,
                  const Tensor &positions,
                  double rope_theta) {
    size_t key = hash_combine(q, k, positions, rope_theta);
    std::shared_ptr<vision::VisionDescriptor> descriptor;
    {
        auto device__ = context::getDevice();
        auto &cache__ = vision::caches.getCache(device__);
        descriptor = cache__.get(key).value_or(nullptr);
        if (!descriptor) {
            descriptor = std::make_shared<vision::VisionDescriptor>(nullptr);
            INFINICORE_CHECK_ERROR(infiniopCreateErnie45VisionRopeDescriptor(
                context::getInfiniopHandle(device__),
                &descriptor->desc,
                q->desc(), k->desc(), positions->desc(), rope_theta));
            cache__.put(key, descriptor);
        }
    }
    INFINIOP_WORKSPACE_TENSOR(workspace, Ernie45VisionRope, descriptor);
    return new VisionPlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(positions)};
}

void run_vision(void *planned_meta) {
    auto *p = reinterpret_cast<VisionPlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(
        infiniopErnie45VisionRope(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->q->data(),
            p->k->data(),
            p->positions->data(),
            context::getStream()));
}

void cleanup_vision(void **planned_meta_ptr) {
    delete *reinterpret_cast<VisionPlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Ernie45VisionRoPE, &plan_vision, &run_vision, &cleanup_vision);

} // namespace vision_op

} // namespace infinicore::op::ernie45_rope_impl::infiniop
