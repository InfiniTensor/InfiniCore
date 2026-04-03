#ifdef ENABLE_ATEN
#include "infinicore/ops/simple_gla_decode_step.hpp"

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

void simple_gla_decode_step_nvidia_impl(Tensor &out, Tensor &state, const Tensor &q, const Tensor &k,
                                        const Tensor &v, const Tensor &g_gamma, float scale) {
    auto aq = infinicore::adaptor::to_aten_tensor(q);
    auto ak = infinicore::adaptor::to_aten_tensor(k);
    auto av = infinicore::adaptor::to_aten_tensor(v);
    auto ag = infinicore::adaptor::to_aten_tensor(g_gamma);
    auto aout = infinicore::adaptor::to_aten_tensor(out);
    auto aS = infinicore::adaptor::to_aten_tensor(state);

    aq = aq.transpose(1, 2).contiguous();
    ak = ak.transpose(1, 2).contiguous();
    av = av.transpose(1, 2).contiguous();

    auto gate = ag.exp().to(at::kFloat);

    at::Tensor k_t = ak.select(2, 0);
    at::Tensor v_t = av.select(2, 0);
    at::Tensor kv = k_t.unsqueeze(-1).mul(v_t.unsqueeze(-2));
    at::Tensor newS = aS.mul(gate.view({1, -1, 1, 1})).add(kv.to(aS.scalar_type()));
    aS.copy_(newS);

    at::Tensor q_t = aq.select(2, 0).to(at::kFloat).mul(scale);
    at::Tensor o_t = q_t.unsqueeze(-2).matmul(aS).squeeze(-2);
    aout.select(1, 0).copy_(o_t.to(aout.scalar_type()));
}

void simple_gla_decode_step_nvidia_calculate(Tensor &out, Tensor &state, const Tensor &q, const Tensor &k,
                                             const Tensor &v, const Tensor &g_gamma, float scale) {
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    simple_gla_decode_step_nvidia_impl(out, state, q, k, v, g_gamma, scale);
}

static bool register_nvidia = []() {
    SimpleGlaDecodeStep::dispatcher().registerDevice(Device::Type::NVIDIA, &simple_gla_decode_step_nvidia_calculate,
                                                     false);
    return true;
}();

} // namespace
#endif // ENABLE_NVIDIA_API

} // namespace infinicore::op
#endif // ENABLE_ATEN
