#include "infinicore/adaptor/iluvatar_gdn_adaptor.hpp"
#include "infinicore/ops/gdn_qkv_norm.hpp"
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::adaptor::iluvatar_gdn {
namespace {
using Chunk = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                       at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                       at::Tensor &, at::Tensor &, at::Tensor &, double,
                       const std::optional<at::Tensor> &);
Chunk symbol() {
    static Chunk fn = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        const char *paths[] = {std::getenv("INFINICORE_IXFORMER_GDN_SO"), "libixformer.so",
                               "/usr/local/lib/python3.10/site-packages/ixformer/libixformer.so",
                               "/usr/local/lib/python3.12/site-packages/ixformer/libixformer.so"};
        for (auto *path : paths) {
            if (!path || !*path) {
                continue;
            }
            auto *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            if (!handle) {
                continue;
            }
            fn = reinterpret_cast<Chunk>(dlsym(handle,
                                               "_ZN8ixformer5infer22chunk_gated_delta_ruleERN2at6TensorES3_S3_S3_S3_S3_S3_S3_S3_S3_S3_dRKSt8optionalIS2_E"));
            if (fn) {
                return;
            }
            dlclose(handle);
        }
    });
    return fn;
}
} // namespace
std::optional<Tensor> chunk(
    const Tensor &q, const Tensor &k, const Tensor &v,
    const Tensor &g, const Tensor &beta, Tensor state,
    const Tensor &offsets, const Tensor &cpu_offsets,
    const Tensor &initial_indices, const Tensor &final_indices) {
    const auto *disabled = std::getenv("INFINICORE_DISABLE_IXFORMER_GDN");
    if ((disabled && std::string(disabled) == "1") || context::isGraphRecording()
        || q->device().getType() != Device::Type::ILUVATAR
        || q->dtype() != DataType::BF16 || k->dtype() != DataType::BF16
        || v->dtype() != DataType::BF16 || state->dtype() != DataType::BF16
        || q->ndim() != 4 || q->size(0) != 1 || q->size(2) != 16 || q->size(3) != 128
        || v->ndim() != 4 || v->size(0) != 1 || v->size(2) != 32 || v->size(3) != 128
        || cpu_offsets->device().getType() != Device::Type::CPU
        || cpu_offsets->dtype() != DataType::I32 || offsets->dtype() != DataType::I32
        || initial_indices->dtype() != DataType::I32 || final_indices->dtype() != DataType::I32) {
        return std::nullopt;
    }
    // This is a deliberately narrow vendor ABI. Unsupported layouts must
    // reach the native operator before any state gather/scatter is enqueued.
    if (q->shape() != k->shape() || q->size(1) != v->size(1)
        || q->strides()[3] != 1 || k->strides()[3] != 1 || v->strides()[3] != 1
        || state->ndim() != 4 || state->size(1) != 32
        || state->size(2) != 128 || state->size(3) != 128 || !state->is_contiguous()
        || g->shape() != std::vector<size_t>{1, q->size(1), 32} || beta->shape() != g->shape()
        || offsets->ndim() != 1 || initial_indices->ndim() != 1 || final_indices->ndim() != 1
        || offsets->numel() != cpu_offsets->numel()
        || initial_indices->numel() != final_indices->numel()
        || !offsets->is_contiguous() || !initial_indices->is_contiguous() || !final_indices->is_contiguous()) {
        return std::nullopt;
    }
    for (const auto &tensor : {k, v, g, beta, state, offsets, initial_indices, final_indices}) {
        if (tensor->device() != q->device()) {
            return std::nullopt;
        }
    }
    if (cpu_offsets->ndim() != 1 || !cpu_offsets->is_contiguous()
        || cpu_offsets->numel() != initial_indices->numel() + 1) {
        return std::nullopt;
    }
    const auto *lengths = reinterpret_cast<const int32_t *>(cpu_offsets->data());
    size_t max_length = 0;
    if (lengths[0] != 0 || lengths[cpu_offsets->numel() - 1] != static_cast<int32_t>(q->size(1))) {
        throw std::invalid_argument("Iluvatar GDN: CPU offsets do not cover the packed input");
    }
    for (size_t i = 1; i < cpu_offsets->numel(); ++i) {
        if (lengths[i] < lengths[i - 1]) {
            throw std::invalid_argument("Iluvatar GDN: nonmonotonic offsets");
        }
        max_length = std::max(max_length, static_cast<size_t>(lengths[i] - lengths[i - 1]));
    }
    if (max_length < 128) {
        return std::nullopt;
    }
    auto fn = symbol();
    if (!fn) {
        return std::nullopt;
    }
    set_aten_stream_to_infinicore();
    at::Tensor qn, kn, va;
    Tensor normalized_q, normalized_k, packed_v;
    if (std::getenv("INFINICORE_DISABLE_FUSED_GDN_QKV_NORM") == nullptr) {
        normalized_q = Tensor::empty(q->shape(), q->dtype(), q->device());
        normalized_k = Tensor::empty(k->shape(), k->dtype(), k->device());
        packed_v = Tensor::empty(v->shape(), v->dtype(), v->device());
        op::gdn_qkv_norm_(normalized_q, normalized_k, packed_v, q, k, v);
        qn = to_aten_tensor(normalized_q).squeeze(0);
        kn = to_aten_tensor(normalized_k).squeeze(0);
        va = to_aten_tensor(packed_v).squeeze(0);
    } else {
        auto qa = to_aten_tensor(q).squeeze(0).to(at::kFloat);
        auto ka = to_aten_tensor(k).squeeze(0).to(at::kFloat);
        qn = (qa * at::rsqrt(qa.square().sum({-1}, true) + 1e-6)).to(at::kBFloat16);
        kn = (ka * at::rsqrt(ka.square().sum({-1}, true) + 1e-6)).to(at::kBFloat16);
        va = to_aten_tensor(v).squeeze(0).contiguous();
    }
    auto ga = to_aten_tensor(g).squeeze(0).to(at::kFloat).contiguous();
    auto ba = to_aten_tensor(beta).squeeze(0).to(at::kBFloat16).contiguous();
    auto sa = to_aten_tensor(state);
    auto cu = to_aten_tensor(offsets);
    auto cpu_cu = to_aten_tensor(cpu_offsets);
    auto indices = to_aten_tensor(final_indices);
    // The vendor operator reads and writes one slot per sequence. Gather all
    // source slots before any scatter so copy-on-write slots remain isolated.
    auto source = sa.index_select(0, to_aten_tensor(initial_indices).to(at::kLong));
    sa.index_copy_(0, indices.to(at::kLong), source);
    auto has_initial = at::ones({indices.numel()}, sa.options().dtype(at::kInt));
    auto output = Tensor::empty(v->shape(), v->dtype(), v->device());
    auto oa = to_aten_tensor(output).squeeze(0);
    fn(qn, kn, va, oa, ga, ba, sa, has_initial, cu, cpu_cu, indices,
       0.08838834764831845, std::nullopt);
    return output;
}
} // namespace infinicore::adaptor::iluvatar_gdn
#else
namespace infinicore::adaptor::iluvatar_gdn {
std::optional<Tensor> chunk(const Tensor &, const Tensor &, const Tensor &,
                            const Tensor &, const Tensor &, Tensor, const Tensor &, const Tensor &,
                            const Tensor &, const Tensor &) { return std::nullopt; }
} // namespace infinicore::adaptor::iluvatar_gdn
#endif
