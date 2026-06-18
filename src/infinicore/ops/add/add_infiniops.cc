#include "infinicore/ops/add.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

namespace infinicore::op::add_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta c, a, b;
    graph::GraphTensor c_tensor, a_tensor, b_tensor;
};

} // namespace

void *plan(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_ASSERT(c->device().getType() == Device::Type::NVIDIA);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);

    return new PlannedMeta{
        TensorMeta(c),
        TensorMeta(a),
        TensorMeta(b),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    infini::ops::Operator<infini::ops::Add>::Call(
        handle,
        config,
        planned->a.tensor(planned->a_tensor),
        planned->b.tensor(planned->b_tensor),
        planned->c.tensor(planned->c_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    Add::plan_dispatcher().registerDevice(Device::Type::NVIDIA, &plan);
    Add::run_dispatcher().registerDevice(Device::Type::NVIDIA, &run);
    Add::cleanup_dispatcher().registerDevice(Device::Type::NVIDIA, &cleanup);
    return true;
}();

} // namespace infinicore::op::add_impl::infiniops
#endif
