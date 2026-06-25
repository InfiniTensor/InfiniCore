#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/gptq_marlin_gemm.hpp"
#include <infiniop.h>

namespace infinicore::op::gptq_marlin_gemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, GptqMarlinGemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, a, b, b_scales, global_scales, b_zeros, g_idx, perm;
    int64_t b_q_type_id;
    bool is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float;
};

void *plan(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    size_t seed = hash_combine(out, a, b, b_scales, global_scales, b_zeros, g_idx, perm);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, GptqMarlinGemm,
        seed,
        out->desc(), a->desc(),
        b->desc(), b_scales->desc(), global_scales->desc(), b_zeros->desc(), g_idx->desc(), perm->desc());

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetGptqMarlinGemmWorkspaceSize(descriptor->desc, &workspace_size));
    Tensor workspace = Tensor::zeros({workspace_size}, DataType::U8, context::getDevice());

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(b_scales),
        graph::GraphTensor(global_scales),
        graph::GraphTensor(b_zeros),
        graph::GraphTensor(g_idx),
        graph::GraphTensor(perm),
        b_q_type_id,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    auto optional_data = [](graph::GraphTensor &tensor) -> void * {
        return tensor->numel() == 0 ? nullptr : tensor->data();
    };

    INFINICORE_CHECK_ERROR(infiniopGptqMarlinGemm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->a->data(),
        planned->b->data(),
        planned->b_scales->data(),
        optional_data(planned->global_scales),
        optional_data(planned->b_zeros),
        optional_data(planned->g_idx),
        optional_data(planned->perm),
        planned->b_q_type_id,
        planned->is_k_full,
        planned->use_atomic_add,
        planned->use_fp32_reduce,
        planned->is_zp_float,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(GptqMarlinGemm, &plan, &run, &cleanup);

} // namespace infinicore::op::gptq_marlin_gemm_impl::infiniop

namespace infinicore::op::gptq_marlin_gemm_impl::infiniop {

size_t workspace_size(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm) {
    size_t seed = hash_combine(out, a, b, b_scales, global_scales, b_zeros, g_idx, perm);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, GptqMarlinGemm,
        seed,
        out->desc(), a->desc(),
        b->desc(), b_scales->desc(), global_scales->desc(), b_zeros->desc(), g_idx->desc(), perm->desc());

    size_t size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetGptqMarlinGemmWorkspaceSize(descriptor->desc, &size));
    return size;
}

struct PlannedWithWorkspaceMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, a, b, b_scales, global_scales, b_zeros, g_idx, perm;
    int64_t b_q_type_id;
    bool is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float;
};

void *plan_with_workspace(Tensor workspace, Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    size_t seed = hash_combine(out, a, b, b_scales, global_scales, b_zeros, g_idx, perm);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, GptqMarlinGemm,
        seed,
        out->desc(), a->desc(),
        b->desc(), b_scales->desc(), global_scales->desc(), b_zeros->desc(), g_idx->desc(), perm->desc());

    return new PlannedWithWorkspaceMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(b_scales),
        graph::GraphTensor(global_scales),
        graph::GraphTensor(b_zeros),
        graph::GraphTensor(g_idx),
        graph::GraphTensor(perm),
        b_q_type_id,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float};
}

void run_with_workspace(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedWithWorkspaceMeta *>(planned_meta);
    auto optional_data = [](graph::GraphTensor &tensor) -> void * {
        return tensor->numel() == 0 ? nullptr : tensor->data();
    };

    context::setDeviceMemoryAsync(planned->workspace->data(), 0, planned->workspace->nbytes(), context::getStream());
    INFINICORE_CHECK_ERROR(infiniopGptqMarlinGemm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->a->data(),
        planned->b->data(),
        planned->b_scales->data(),
        optional_data(planned->global_scales),
        optional_data(planned->b_zeros),
        optional_data(planned->g_idx),
        optional_data(planned->perm),
        planned->b_q_type_id,
        planned->is_k_full,
        planned->use_atomic_add,
        planned->use_fp32_reduce,
        planned->is_zp_float,
        context::getStream()));
}

void cleanup_with_workspace(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedWithWorkspaceMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered_with_workspace = []() {
    GptqMarlinGemmWithWorkspace::plan_dispatcher().registerAll(&plan_with_workspace, false);
    GptqMarlinGemmWithWorkspace::run_dispatcher().registerAll(&run_with_workspace, false);
    GptqMarlinGemmWithWorkspace::cleanup_dispatcher().registerAll(&cleanup_with_workspace, false);
    return true;
}();

void direct_with_workspace(Tensor workspace, Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    size_t seed = hash_combine(out, a, b, b_scales, global_scales, b_zeros, g_idx, perm);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, GptqMarlinGemm,
        seed,
        out->desc(), a->desc(),
        b->desc(), b_scales->desc(), global_scales->desc(), b_zeros->desc(), g_idx->desc(), perm->desc());

    auto optional_data = [](Tensor &tensor) -> void * {
        return tensor->numel() == 0 ? nullptr : tensor->data();
    };

    context::setDeviceMemoryAsync(workspace->data(), 0, workspace->nbytes(), context::getStream());
    INFINICORE_CHECK_ERROR(infiniopGptqMarlinGemm(
        descriptor->desc,
        workspace->data(),
        workspace->numel(),
        out->data(),
        a->data(),
        b->data(),
        b_scales->data(),
        optional_data(global_scales),
        optional_data(b_zeros),
        optional_data(g_idx),
        optional_data(perm),
        b_q_type_id,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        context::getStream()));
}

} // namespace infinicore::op::gptq_marlin_gemm_impl::infiniop

namespace infinicore::op {

size_t gptq_marlin_gemm_workspace_size(Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm) {
    return gptq_marlin_gemm_impl::infiniop::workspace_size(out, a, b, b_scales, global_scales, b_zeros, g_idx, perm);
}

void gptq_marlin_gemm_with_workspace_direct_(Tensor workspace, Tensor out, const Tensor &a, const Tensor &b, Tensor &b_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    gptq_marlin_gemm_impl::infiniop::direct_with_workspace(workspace, out, a, b, b_scales, global_scales, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float);
}

} // namespace infinicore::op
