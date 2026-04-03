#ifdef ENABLE_ATEN
#include "infinicore/ops/simple_gla_recurrent_state_append.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace infinicore::op {

#ifdef ENABLE_NVIDIA_API
namespace {

void simple_gla_recurrent_state_append_nvidia(Tensor &state, const Tensor &k_seg, const Tensor &v_seg,
                                              const Tensor &g_gamma) {
    auto ak = infinicore::adaptor::to_aten_tensor(k_seg);
    auto av = infinicore::adaptor::to_aten_tensor(v_seg);
    auto ag = infinicore::adaptor::to_aten_tensor(g_gamma);
    auto aS = infinicore::adaptor::to_aten_tensor(state);

    const int64_t B = ak.size(0);
    const int64_t L = ak.size(1);
    const int64_t H = ak.size(2);
    const int64_t D = ak.size(3);
    INFINICORE_ASSERT(L > 0);

    auto gate = ag.exp().to(at::kFloat).contiguous();
    auto j = at::arange(L, at::TensorOptions().dtype(at::kFloat).device(gate.device()));
    auto pows = (static_cast<float>(L - 1) - j) * 0.5f;
    auto w = at::pow(gate.unsqueeze(0), pows.unsqueeze(1)).to(at::kFloat).contiguous();

    // [B,L,H,D] -> [B,H,L,D] so each (b,h) owns a contiguous [L,D] panel for bmm.
    auto kf = ak.to(at::kFloat).permute({0, 2, 1, 3}).contiguous();
    auto vf = av.to(at::kFloat).permute({0, 2, 1, 3}).contiguous();
    // w[l,h] scales token l, head h; broadcast to [1,H,L,1] on [B,H,L,D].
    auto w_bhl = w.transpose(0, 1).contiguous().view({1, H, L, 1});
    kf.mul_(w_bhl);
    vf.mul_(w_bhl);

    auto ks = kf.view({B * H, L, D});
    auto vs = vf.view({B * H, L, D});
    auto s_inc = at::bmm(ks.transpose(1, 2), vs);

    auto gL = at::pow(gate, static_cast<float>(L)).view({1, H, 1, 1});
    aS.mul_(gL).add_(s_inc.view({B, H, D, D}));
}

void simple_gla_recurrent_state_append_nvidia_calculate(Tensor &state, const Tensor &k_seg, const Tensor &v_seg,
                                                      const Tensor &g_gamma) {
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    simple_gla_recurrent_state_append_nvidia(state, k_seg, v_seg, g_gamma);
}

static bool register_nvidia = []() {
    SimpleGlaRecurrentStateAppend::dispatcher().registerDevice(Device::Type::NVIDIA,
                                                               &simple_gla_recurrent_state_append_nvidia_calculate,
                                                               false);
    return true;
}();

} // namespace
#endif // ENABLE_NVIDIA_API

} // namespace infinicore::op
#endif // ENABLE_ATEN
