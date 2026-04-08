#include "infinicore/ops/simple_gla_recurrent_state_append.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/simple_gla_decode_step.hpp"

namespace infinicore::op {

namespace {

void simple_gla_recurrent_state_append_cpu(Tensor &state, const Tensor &k_seg, const Tensor &v_seg,
                                           const Tensor &g_gamma) {
    const auto &sh = k_seg->shape();
    const size_t B = sh[0];
    const size_t L = sh[1];
    const size_t H = sh[2];
    const size_t D = sh[3];
    INFINICORE_ASSERT(v_seg->shape() == sh);

    auto q_zero = Tensor::zeros({B, 1, H, D}, k_seg->dtype(), k_seg->device());
    auto out = Tensor::empty({B, 1, H, D}, k_seg->dtype(), k_seg->device());
    for (size_t t = 0; t < L; ++t) {
        auto kt = k_seg->narrow({{1, t, 1}});
        auto vt = v_seg->narrow({{1, t, 1}});
        SimpleGlaDecodeStep::execute(out, state, q_zero, kt, vt, g_gamma, 1.0f);
    }
}

static bool register_cpu = []() {
    SimpleGlaRecurrentStateAppend::dispatcher().registerDevice(Device::Type::CPU, &simple_gla_recurrent_state_append_cpu,
                                                               false);
    return true;
}();

} // namespace

common::OpDispatcher<SimpleGlaRecurrentStateAppend::schema> &SimpleGlaRecurrentStateAppend::dispatcher() {
    static common::OpDispatcher<schema> dispatcher_;
    return dispatcher_;
}

void SimpleGlaRecurrentStateAppend::execute(Tensor &state, const Tensor &k_seg, const Tensor &v_seg,
                                            const Tensor &g_gamma) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(state, k_seg, v_seg, g_gamma);

    const auto &sh = k_seg->shape();
    INFINICORE_ASSERT(sh.size() == 4);
    const size_t B = sh[0];
    const size_t L = sh[1];
    const size_t H = sh[2];
    const size_t D = sh[3];
    INFINICORE_ASSERT(v_seg->shape() == sh);
    INFINICORE_ASSERT(state->shape() == Shape({B, H, D, D}));
    INFINICORE_ASSERT(state->dtype() == DataType::F32);
    INFINICORE_ASSERT(g_gamma->shape() == Shape({H}));
    INFINICORE_ASSERT(state->is_contiguous());

    if (L == 0) {
        return;
    }

    infinicore::context::setDevice(state->device());
    auto dev = infinicore::context::getDevice().getType();
    auto fn = SimpleGlaRecurrentStateAppend::dispatcher().lookup(dev);
    if (fn == nullptr) {
        throw std::runtime_error("simple_gla_recurrent_state_append_segment: no implementation for device type " + std::to_string(static_cast<int>(dev)));
    }
    fn(state, k_seg->contiguous(), v_seg->contiguous(), g_gamma);
}

void simple_gla_recurrent_state_append_segment(Tensor &state, const Tensor &k_seg, const Tensor &v_seg,
                                               const Tensor &g_gamma) {
    SimpleGlaRecurrentStateAppend::execute(state, k_seg, v_seg, g_gamma);
}

} // namespace infinicore::op
