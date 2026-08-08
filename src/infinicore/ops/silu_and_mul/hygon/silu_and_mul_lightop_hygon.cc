#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/silu_and_mul.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/lightop_adaptor.hpp"

#include <c10/hip/HIPGuard.h>

namespace infinicore::op::silu_and_mul_impl::lightop_hygon {

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor x;
};

void *plan(Tensor out, const Tensor &x) {
    infinicore::adaptor::lightop::preload_silu_and_mul();
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    Tensor x_work_ic = p->x->is_contiguous() ? Tensor(p->x) : p->x->contiguous();

    auto out = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto x = infinicore::adaptor::to_aten_tensor(x_work_ic);

    infinicore::adaptor::lightop::fuse_silu_and_mul(x, out);

    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    if (!infinicore::adaptor::lightop::available()) {
        return false;
    }
    SiluAndMul::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    SiluAndMul::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    SiluAndMul::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::silu_and_mul_impl::lightop_hygon

#endif // ENABLE_HYGON_API && ENABLE_ATEN
