#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/awq_marlin_gemm.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::awq_marlin_gemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, AwqMarlinGemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a, b, b_bias, b_scales, a_scales, global_scales, b_zeros, g_idx, perm;
    int64_t b_q_type_id;
    bool is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float;
};

void *plan(Tensor c, const Tensor &a, const Tensor &b, Tensor &b_bias, Tensor &b_scales, Tensor &a_scales, Tensor &global_scales, Tensor &b_zeros, Tensor &g_idx, Tensor &perm, int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
    size_t seed = hash_combine(c, a, b, b_bias, b_scales, a_scales, global_scales, b_zeros, g_idx, perm);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, AwqMarlinGemm,
        seed,
        c->desc(), a->desc(),
        b->desc(), b_bias->desc(), b_scales->desc(), a_scales->desc(), global_scales->desc(), b_zeros->desc(), g_idx->desc(), perm->desc());

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAwqMarlinGemmWorkspaceSize(descriptor->desc, &workspace_size));
    Tensor workspace = Tensor::zeros({workspace_size}, DataType::U8, context::getDevice());

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(b_bias),
        graph::GraphTensor(b_scales),
        graph::GraphTensor(a_scales),
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

    INFINICORE_CHECK_ERROR(infiniopAwqMarlinGemm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->c->data(),
        planned->a->data(),
        planned->b->data(),
        optional_data(planned->b_bias),
        planned->b_scales->data(),
        optional_data(planned->a_scales),
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

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(AwqMarlinGemm, &plan, &run, &cleanup);

} // namespace infinicore::op::awq_marlin_gemm_impl::infiniop
