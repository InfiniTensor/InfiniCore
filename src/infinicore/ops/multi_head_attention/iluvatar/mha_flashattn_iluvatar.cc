#if defined(ENABLE_ILUVATAR_FLASH_ATTN)
#include "infinicore/ops/mha.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <cstdlib>

#include <stdexcept>
#include <string>

namespace infinicore::op::mha_impl::flashattn {

struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    bool is_causal;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {
    auto *meta = new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        is_causal};
    return meta;
}

void run(void *planned_meta) {
    infinicore::adaptor::set_aten_stream_to_infinicore();
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto out = std::optional<at::Tensor>(out_work);

    auto alibi_slopes = p->alibi_slopes ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes)) : std::nullopt;
    auto scale = p->scale;
    auto is_causal = p->is_causal;

    if (alibi_slopes.has_value()) {
        throw std::runtime_error("Iluvatar MHA does not support alibi_slopes");
    }

    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) {
        throw std::runtime_error("Iluvatar MHA expects rank-4 q, k and v tensors");
    }

    // CoreX's BF16 vision kernel is substantially faster with aligned heads.
    // Padding changes neither the dot product nor the original softmax scale.
    // Keep this exact-shape policy local to the verified Qwen3.5 vision path.
    const char *disable_padding = std::getenv("INFINICORE_DISABLE_VISION_HEAD_PADDING");
    const bool pad_vision_heads = !(disable_padding && std::string(disable_padding) == "1") && q.scalar_type() == at::kBFloat16 && q.size(3) == 72 && q.size(2) == 16 && k.size(2) == 16 && v.size(2) == 16 && k.size(3) == 72 && v.size(3) == 72 && !is_causal;
    if (pad_vision_heads) {
        q = at::constant_pad_nd(q, {0, 56}, 0);
        k = at::constant_pad_nd(k, {0, 56}, 0);
        v = at::constant_pad_nd(v, {0, 56}, 0);
    }

    const int64_t num_heads = q.size(2);
    const int64_t num_kv_heads = k.size(2);
    auto q_sdpa = q.permute({0, 2, 1, 3});
    auto k_sdpa = k.permute({0, 2, 1, 3});
    auto v_sdpa = v.permute({0, 2, 1, 3});
    if (num_heads != num_kv_heads) {
        if (num_heads % num_kv_heads != 0) {
            throw std::runtime_error("Iluvatar MHA requires num_heads to be divisible by num_kv_heads");
        }
        const int64_t groups = num_heads / num_kv_heads;
        k_sdpa = k_sdpa.repeat_interleave(groups, 1);
        v_sdpa = v_sdpa.repeat_interleave(groups, 1);
    }

    auto result = at::scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        std::nullopt,
        0.0,
        is_causal,
        std::optional<double>(static_cast<double>(scale)));
    auto result_bshd = result.permute({0, 2, 1, 3});
    if (pad_vision_heads) {
        result_bshd = result_bshd.narrow(3, 0, 72);
    }
    out_work.copy_(result_bshd);

    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MultiheadAttention, &plan, &run, &cleanup);

} // namespace infinicore::op::mha_impl::flashattn
#endif
