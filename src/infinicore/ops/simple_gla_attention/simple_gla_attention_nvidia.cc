#ifdef ENABLE_ATEN
#include "infinicore/ops/simple_gla_attention.hpp"

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

void simple_gla_attention_nvidia_impl(Tensor & out, const Tensor &q, const Tensor &k,
                                     const Tensor &v, const Tensor &g_gamma, float scale) {
    auto aq = infinicore::adaptor::to_aten_tensor(q);
    auto ak = infinicore::adaptor::to_aten_tensor(k);
    auto av = infinicore::adaptor::to_aten_tensor(v);
    auto ag = infinicore::adaptor::to_aten_tensor(g_gamma);
    auto aout = infinicore::adaptor::to_aten_tensor(out);

    const int64_t B = aq.size(0);
    const int64_t T = aq.size(1);
    const int64_t H = aq.size(2);
    const int64_t D = aq.size(3);

    // (B, T, H, D) -> (B, H, T, D) for step-by-step
    aq = aq.transpose(1, 2).contiguous();   // (B, H, T, D)
    ak = ak.transpose(1, 2).contiguous();
    av = av.transpose(1, 2).contiguous();

    // Accumulate in float32 for stability (match CPU behavior)
    auto gate = ag.exp().to(at::kFloat);   // (H,)
    auto S = at::zeros({B, H, D, D}, aq.options().dtype(at::kFloat));

    for (int64_t i = 0; i < T; ++i) {
        // k_t, v_t: (B, H, D)
        at::Tensor k_t = ak.select(2, i);
        at::Tensor v_t = av.select(2, i);
        // kv = outer(k_t, v_t): (B, H, D, D)
        at::Tensor kv = k_t.unsqueeze(-1).mul(v_t.unsqueeze(-2));
        S = S.mul(gate.view({1, -1, 1, 1})).add(kv.to(S.dtype()));

        // q_t: (B, H, D), scaled
        at::Tensor q_t = aq.select(2, i).to(at::kFloat).mul(scale);
        // o_t = (q_t @ S): for each (b,h), o[b,h,:] = q_t[b,h,:] @ S[b,h,:,:] -> (B, H, D)
        at::Tensor o_t = q_t.unsqueeze(-2).matmul(S).squeeze(-2);
        aout.select(1, i).copy_(o_t.to(aout.dtype()));
    }
}

void simple_gla_attention_nvidia_calculate(Tensor & out, const Tensor &q, const Tensor &k,
                                           const Tensor &v, const Tensor &g_gamma, float scale) {
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    simple_gla_attention_nvidia_impl(out, q, k, v, g_gamma, scale);
}

static bool register_nvidia = []() {
    SimpleGlaAttention::dispatcher().registerDevice(Device::Type::NVIDIA,
                                                    &simple_gla_attention_nvidia_calculate, false);
    return true;
}();

} // namespace
#endif // ENABLE_NVIDIA_API

} // namespace infinicore::op
#endif // ENABLE_ATEN
