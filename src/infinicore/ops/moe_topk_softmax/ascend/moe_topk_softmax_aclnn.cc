#if defined(ENABLE_ASCEND_API)

#include "infinicore/context/context.hpp"
#include "infinicore/ops/moe_topk_softmax.hpp"
#include "native/ascend/workspace_pool_.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_moe_gating_top_k_softmax_v2.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::moe_topk_softmax_impl::aclnn {
namespace {

aclDataType to_acl_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::F16:
        return ACL_FLOAT16;
    case DataType::BF16:
        return ACL_BF16;
    case DataType::F32:
        return ACL_FLOAT;
    case DataType::I32:
        return ACL_INT32;
    default:
        throw std::runtime_error("[moe_topk_softmax/ascend] unsupported dtype");
    }
}

std::vector<int64_t> to_i64(const Shape &values) {
    std::vector<int64_t> result;
    result.reserve(values.size());
    for (const auto value : values) {
        result.push_back(static_cast<int64_t>(value));
    }
    return result;
}

std::vector<int64_t> to_i64(const Strides &values) {
    std::vector<int64_t> result;
    result.reserve(values.size());
    for (const auto value : values) {
        result.push_back(static_cast<int64_t>(value));
    }
    return result;
}

aclTensor *to_acl_tensor(const Tensor &tensor) {
    const auto dims = to_i64(tensor->shape());
    const auto strides = to_i64(tensor->strides());
    return aclCreateTensor(
        dims.data(), dims.size(), to_acl_dtype(tensor->dtype()), strides.data(),
        0, ACL_FORMAT_ND, dims.data(), dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(tensor->data())));
}

void check_aclnn(aclnnStatus status, const char *operation) {
    if (status == ACL_SUCCESS) {
        return;
    }
    const char *message = aclGetRecentErrMsg();
    throw std::runtime_error(
        std::string("[moe_topk_softmax/ascend] ") + operation + " failed: "
        + std::to_string(status) + ", msg: "
        + (message == nullptr ? "(null)" : message));
}

struct PlannedMeta {
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_indices;
    graph::GraphTensor gating_output;
    graph::GraphTensor native_weights;
    bool renormalize;
};

} // namespace

void *plan(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &gating_output,
           const Tensor &correction_bias,
           const bool renormalize,
           const float moe_softcapping) {
    if (correction_bias) {
        throw std::runtime_error(
            "[moe_topk_softmax/ascend] correction bias is not supported");
    }
    if (moe_softcapping != 0.0f) {
        throw std::runtime_error(
            "[moe_topk_softmax/ascend] softcapping is not supported");
    }
    if (topk_weights->dtype() != DataType::F32
        || topk_indices->dtype() != DataType::I32) {
        throw std::runtime_error(
            "[moe_topk_softmax/ascend] outputs must be float32 and int32");
    }
    if (gating_output->dtype() != DataType::F16
        && gating_output->dtype() != DataType::BF16
        && gating_output->dtype() != DataType::F32) {
        throw std::runtime_error(
            "[moe_topk_softmax/ascend] logits must be float16, bfloat16, or float32");
    }

    Tensor native_weights = gating_output->dtype() == DataType::F32
                              ? topk_weights
                              : Tensor::empty(topk_weights->shape(),
                                              gating_output->dtype(),
                                              gating_output->device());
    return new PlannedMeta{
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(gating_output),
        graph::GraphTensor(native_weights),
        renormalize};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    context::setDevice(p->gating_output->device());

    aclTensor *logits_acl = to_acl_tensor(p->gating_output);
    aclTensor *native_weights_acl = to_acl_tensor(p->native_weights);
    aclTensor *indices_acl = to_acl_tensor(p->topk_indices);
    const int64_t topk = static_cast<int64_t>(p->topk_indices->size(1));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(
        logits_acl, nullptr, topk, p->renormalize ? 1 : 0, false,
        native_weights_acl, indices_acl, nullptr, &workspace_size, &executor);
    check_aclnn(status, "aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize");

    const auto stream = static_cast<aclrtStream>(context::getStream());
    void *workspace = workspace_size == 0
                        ? nullptr
                        : infini::ops::ascend::GetWorkspacePool()
                              .Ensure(stream, workspace_size, "moe_topk_v2")
                              .buf;
    status = aclnnMoeGatingTopKSoftmaxV2(
        workspace, workspace_size, executor, stream);
    check_aclnn(status, "aclnnMoeGatingTopKSoftmaxV2");

    if (p->native_weights->dtype() != DataType::F32) {
        aclTensor *weights_acl = to_acl_tensor(p->topk_weights);
        uint64_t cast_workspace_size = 0;
        aclOpExecutor *cast_executor = nullptr;
        status = aclnnCastGetWorkspaceSize(
            native_weights_acl, ACL_FLOAT, weights_acl,
            &cast_workspace_size, &cast_executor);
        check_aclnn(status, "aclnnCastGetWorkspaceSize");
        void *cast_workspace = cast_workspace_size == 0
                                 ? nullptr
                                 : infini::ops::ascend::GetWorkspacePool()
                                       .Ensure(stream, cast_workspace_size,
                                               "moe_topk_cast")
                                       .buf;
        status = aclnnCast(
            cast_workspace, cast_workspace_size, cast_executor, stream);
        aclDestroyTensor(weights_acl);
        check_aclnn(status, "aclnnCast");
    }

    aclDestroyTensor(logits_acl);
    aclDestroyTensor(native_weights_acl);
    aclDestroyTensor(indices_acl);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MoeTopkSoftmax::plan_dispatcher().registerDevice(Device::Type::ASCEND, &plan);
    MoeTopkSoftmax::run_dispatcher().registerDevice(Device::Type::ASCEND, &run);
    MoeTopkSoftmax::cleanup_dispatcher().registerDevice(
        Device::Type::ASCEND, &cleanup);
    return true;
}();

} // namespace infinicore::op::moe_topk_softmax_impl::aclnn

#endif
