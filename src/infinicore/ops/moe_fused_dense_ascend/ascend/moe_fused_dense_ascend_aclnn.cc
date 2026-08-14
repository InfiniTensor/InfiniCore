#if defined(ENABLE_ASCEND_API)

#include "infinicore/context/context.hpp"
#include "infinicore/ops/moe_fused_dense_ascend.hpp"
#include "native/ascend/workspace_pool_.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_ge_scalar.h>
#include <aclnnop/aclnn_grouped_matmul_v4.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_moe_token_unpermute.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_swi_glu.h>

#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::moe_fused_dense_ascend_impl::aclnn {
namespace {

using RouteGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    bool,
    int64_t,
    const aclIntArray *,
    int64_t,
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    uint64_t *,
    aclOpExecutor **);

using RouteRunFn = aclnnStatus (*)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

struct VendorApi {
    void *handle;
    RouteGetWorkspaceSizeFn get_workspace_size;
    RouteRunFn run;
};

void configure_custom_opp_path() {
    const char *override_path = std::getenv("INFINICORE_ASCEND_CUSTOM_OPP_PATH");
    const std::string opp_path = override_path != nullptr
                                   ? override_path
                                   : "/vllm-workspace/vllm-ascend/"
                                     "vllm_ascend/_cann_ops_custom/"
                                     "vendors/vllm-ascend";
    const char *current = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    const std::string current_path = current != nullptr ? current : "";
    if (current_path.find(opp_path) == std::string::npos) {
        const std::string combined = current_path.empty()
                                       ? opp_path
                                       : opp_path + ":" + current_path;
        setenv("ASCEND_CUSTOM_OPP_PATH", combined.c_str(), 1);
    }
}

VendorApi load_vendor_api() {
    configure_custom_opp_path();
    std::vector<std::string> candidates;
    if (const char *override_path = std::getenv("INFINICORE_ASCEND_MOE_VENDOR_SO")) {
        candidates.emplace_back(override_path);
    }
    candidates.emplace_back(
        "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/"
        "vendors/vllm-ascend/op_api/lib/libcust_opapi.so");
    candidates.emplace_back(
        "/usr/local/lib/python3.11/site-packages/vllm_ascend/"
        "_cann_ops_custom/vendors/vllm-ascend/op_api/lib/"
        "libcust_opapi.so");

    std::string errors;
    for (const auto &path : candidates) {
        dlerror();
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle == nullptr) {
            const char *error = dlerror();
            errors += "\n  " + path + ": "
                    + (error != nullptr ? error : "unknown dlopen error");
            continue;
        }
        auto get_workspace_size = reinterpret_cast<RouteGetWorkspaceSizeFn>(
            dlsym(handle,
                  "aclnnMoeInitRoutingCustomGetWorkspaceSize"));
        auto run = reinterpret_cast<RouteRunFn>(
            dlsym(handle, "aclnnMoeInitRoutingCustom"));
        if (get_workspace_size != nullptr && run != nullptr) {
            return {handle, get_workspace_size, run};
        }
        errors += "\n  " + path + ": required symbols are missing";
        dlclose(handle);
    }
    throw std::runtime_error(
        "[moe_fused_dense_ascend] unable to load vLLM-Ascend "
        "MoeInitRoutingCustom. Set INFINICORE_ASCEND_MOE_VENDOR_SO "
        "and INFINICORE_ASCEND_CUSTOM_OPP_PATH if installed elsewhere."
        + errors);
}

VendorApi &vendor_api() {
    static VendorApi api = load_vendor_api();
    return api;
}

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
    case DataType::I64:
        return ACL_INT64;
    case DataType::BOOL:
        return ACL_BOOL;
    default:
        throw std::runtime_error(
            "[moe_fused_dense_ascend] unsupported dtype");
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

aclTensor *to_acl_tensor(const graph::GraphTensor &tensor) {
    const auto dims = to_i64(tensor->shape());
    const auto strides = to_i64(tensor->strides());
    return aclCreateTensor(
        dims.data(), dims.size(), to_acl_dtype(tensor->dtype()),
        strides.data(), 0, ACL_FORMAT_ND, dims.data(), dims.size(),
        const_cast<void *>(
            reinterpret_cast<const void *>(tensor->data())));
}

void check_aclnn(aclnnStatus status, const char *operation) {
    if (status == ACL_SUCCESS) {
        return;
    }
    const char *message = aclGetRecentErrMsg();
    throw std::runtime_error(
        std::string("[moe_fused_dense_ascend] ") + operation
        + " failed: " + std::to_string(status) + ", msg: "
        + (message == nullptr ? "(null)" : message));
}

void run_executor(const char *key,
                  uint64_t workspace_size,
                  aclOpExecutor *executor,
                  aclnnStatus (*run)(void *,
                                     uint64_t,
                                     aclOpExecutor *,
                                     aclrtStream),
                  aclrtStream stream) {
    void *workspace = workspace_size == 0
                        ? nullptr
                        : infini::ops::ascend::GetWorkspacePool()
                              .Ensure(stream, workspace_size, key)
                              .buf;
    check_aclnn(
        run(workspace, workspace_size, executor, stream), key);
}

class ArenaLayout {
public:
    size_t add(const Shape &shape, DataType dtype) {
        constexpr size_t alignment = 512;
        size_ = (size_ + alignment - 1) / alignment * alignment;
        const size_t offset = size_;
        size_t elements = 1;
        for (const auto dim : shape) {
            elements *= dim;
        }
        size_ += elements * dsize(dtype);
        return offset;
    }

    size_t size() const {
        return size_;
    }

private:
    size_t size_ = 0;
};

Tensor arena_tensor(void *base,
                    size_t offset,
                    const Shape &shape,
                    DataType dtype,
                    const Device &device) {
    return Tensor::from_blob(
        static_cast<std::byte *>(base) + offset, shape, dtype, device);
}

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w13;
    graph::GraphTensor w2;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_ids;
    graph::GraphTensor expanded_hidden;
    graph::GraphTensor expanded_row_ids;
    graph::GraphTensor expert_counts;
    graph::GraphTensor expanded_scale;
    graph::GraphTensor gate_up;
    graph::GraphTensor activated;
    graph::GraphTensor expert_output;
    graph::GraphTensor local_mask;
    graph::GraphTensor masked_weights;
    graph::GraphTensor local_lower_mask;
    graph::GraphTensor local_upper_mask;
    graph::GraphTensor absolute_row_ids;
    size_t global_num_experts;
    size_t local_expert_start;
    size_t local_num_experts;
};

} // namespace

void *plan(Tensor output,
           const Tensor &hidden_states,
           const Tensor &w13,
           const Tensor &w2,
           const Tensor &topk_weights,
           const Tensor &topk_ids,
           size_t global_num_experts,
           size_t local_expert_start,
           size_t local_num_experts) {
    if (hidden_states->shape().size() != 2
        || topk_ids->shape().size() != 2
        || topk_weights->shape() != topk_ids->shape()) {
        throw std::runtime_error(
            "[moe_fused_dense_ascend] expected hidden [T,H] and "
            "matching top-k tensors [T,K]");
    }
    const size_t tokens = hidden_states->size(0);
    const size_t hidden = hidden_states->size(1);
    const size_t topk = topk_ids->size(1);
    const size_t pairs = tokens * topk;
    if (output->shape() != hidden_states->shape()
        || topk_ids->dtype() != DataType::I32
        || topk_weights->dtype() != DataType::F32
        || (hidden_states->dtype() != DataType::F16
            && hidden_states->dtype() != DataType::BF16)
        || !hidden_states->is_contiguous()
        || !output->is_contiguous()
        || !w13->is_contiguous()
        || !w2->is_contiguous()) {
        throw std::runtime_error(
            "[moe_fused_dense_ascend] unsupported dtype, shape, or "
            "non-contiguous runtime tensor");
    }
    if (w13->shape().size() != 3
        || w13->size(0) != local_num_experts
        || w13->size(1) != hidden
        || w13->size(2) % 2 != 0) {
        throw std::runtime_error(
            "[moe_fused_dense_ascend] w13 must be [E,H,2I]");
    }
    const size_t intermediate = w13->size(2) / 2;
    if (w2->shape()
            != Shape{local_num_experts, intermediate, hidden}
        || local_expert_start + local_num_experts
               > global_num_experts) {
        throw std::runtime_error(
            "[moe_fused_dense_ascend] w2 or expert placement mismatch");
    }

    const auto dtype = hidden_states->dtype();
    const auto device = hidden_states->device();
    const Shape expanded_hidden_shape{pairs, hidden};
    const Shape pairs_shape{pairs};
    const Shape counts_shape{local_num_experts};
    const Shape gate_up_shape{pairs, intermediate * 2};
    const Shape activated_shape{pairs, intermediate};

    ArenaLayout layout;
    const auto expanded_hidden_offset = layout.add(expanded_hidden_shape, dtype);
    const auto expanded_row_ids_offset = layout.add(pairs_shape, DataType::I32);
    const auto expert_counts_offset = layout.add(counts_shape, DataType::I64);
    const auto expanded_scale_offset = layout.add(pairs_shape, DataType::F32);
    const auto gate_up_offset = layout.add(gate_up_shape, dtype);
    const auto activated_offset = layout.add(activated_shape, dtype);
    const auto expert_output_offset = layout.add(expanded_hidden_shape, dtype);
    const auto local_mask_offset = layout.add(topk_ids->shape(), DataType::BOOL);
    const auto masked_weights_offset = layout.add(topk_weights->shape(), DataType::F32);
    const auto local_lower_mask_offset = layout.add(topk_ids->shape(), DataType::BOOL);
    const auto local_upper_mask_offset = layout.add(topk_ids->shape(), DataType::BOOL);
    const auto absolute_row_ids_offset = layout.add(pairs_shape, DataType::I32);

    context::setDevice(device);
    const auto stream = static_cast<aclrtStream>(context::getStream());
    void *arena = infini::ops::ascend::GetWorkspacePool()
                      .Ensure(stream, layout.size(),
                              "moe_fused_intermediates")
                      .buf;

    return new PlannedMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(hidden_states),
        graph::GraphTensor(w13),
        graph::GraphTensor(w2),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_ids),
        graph::GraphTensor(arena_tensor(
            arena, expanded_hidden_offset, expanded_hidden_shape, dtype,
            device)),
        graph::GraphTensor(arena_tensor(
            arena, expanded_row_ids_offset, pairs_shape, DataType::I32,
            device)),
        graph::GraphTensor(arena_tensor(
            arena, expert_counts_offset, counts_shape, DataType::I64,
            device)),
        graph::GraphTensor(arena_tensor(
            arena, expanded_scale_offset, pairs_shape, DataType::F32,
            device)),
        graph::GraphTensor(arena_tensor(
            arena, gate_up_offset, gate_up_shape, dtype, device)),
        graph::GraphTensor(arena_tensor(
            arena, activated_offset, activated_shape, dtype, device)),
        graph::GraphTensor(arena_tensor(
            arena, expert_output_offset, expanded_hidden_shape, dtype,
            device)),
        graph::GraphTensor(arena_tensor(
            arena, local_mask_offset, topk_ids->shape(), DataType::BOOL,
            device)),

        graph::GraphTensor(arena_tensor(
            arena, masked_weights_offset, topk_weights->shape(),
            DataType::F32, device)),
        graph::GraphTensor(arena_tensor(
            arena, local_lower_mask_offset, topk_ids->shape(),
            DataType::BOOL, device)),
        graph::GraphTensor(arena_tensor(
            arena, local_upper_mask_offset, topk_ids->shape(),
            DataType::BOOL, device)),
        graph::GraphTensor(arena_tensor(
            arena, absolute_row_ids_offset, pairs_shape, DataType::I32,
            device)),
        global_num_experts,
        local_expert_start,
        local_num_experts};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    context::setDevice(p->hidden_states->device());
    const auto stream = static_cast<aclrtStream>(context::getStream());
    const int64_t pairs = static_cast<int64_t>(p->expanded_hidden->size(0));
    const std::vector<int64_t> expert_range{
        static_cast<int64_t>(p->local_expert_start),
        static_cast<int64_t>(
            p->local_expert_start + p->local_num_experts)};
    aclIntArray *range_acl = aclCreateIntArray(
        expert_range.data(), expert_range.size());

    aclTensor *hidden_acl = to_acl_tensor(p->hidden_states);
    aclTensor *topk_ids_acl = to_acl_tensor(p->topk_ids);
    aclTensor *expanded_hidden_acl = to_acl_tensor(p->expanded_hidden);
    aclTensor *expanded_row_ids_acl = to_acl_tensor(p->expanded_row_ids);
    aclTensor *expert_counts_acl = to_acl_tensor(p->expert_counts);
    aclTensor *expanded_scale_acl = to_acl_tensor(p->expanded_scale);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto &route = vendor_api();
    check_aclnn(
        route.get_workspace_size(
            hidden_acl, topk_ids_acl, nullptr, nullptr, pairs, -1,
            static_cast<int64_t>(p->global_num_experts), 0, 1, true,
            -1, range_acl, 0, expanded_hidden_acl,
            expanded_row_ids_acl, expert_counts_acl,
            expanded_scale_acl, &workspace_size, &executor),
        "aclnnMoeInitRoutingCustomGetWorkspaceSize");
    void *workspace = workspace_size == 0
                        ? nullptr
                        : infini::ops::ascend::GetWorkspacePool()
                              .Ensure(stream, workspace_size,
                                      "moe_fused_ascend")
                              .buf;
    check_aclnn(
        route.run(workspace, workspace_size, executor, stream),
        "aclnnMoeInitRoutingCustom");

    aclTensor *w13_acl = to_acl_tensor(p->w13);
    aclTensor *gate_up_acl = to_acl_tensor(p->gate_up);
    aclTensor *gmm_x_items[]{expanded_hidden_acl};
    aclTensor *gmm_w13_items[]{w13_acl};
    aclTensor *gmm_gate_up_items[]{gate_up_acl};
    aclTensorList *gmm_x = aclCreateTensorList(gmm_x_items, 1);
    aclTensorList *gmm_w13 = aclCreateTensorList(gmm_w13_items, 1);
    aclTensorList *gmm_gate_up = aclCreateTensorList(gmm_gate_up_items, 1);
    workspace_size = 0;
    executor = nullptr;
    check_aclnn(
        aclnnGroupedMatmulV4GetWorkspaceSize(
            gmm_x, gmm_w13, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, expert_counts_acl, nullptr, nullptr,
            nullptr, 2, 0, 1, 0, gmm_gate_up, nullptr, nullptr,
            &workspace_size, &executor),
        "aclnnGroupedMatmulV4GetWorkspaceSize(w13)");
    run_executor(
        "moe_fused_ascend", workspace_size, executor,
        &aclnnGroupedMatmulV4, stream);

    aclTensor *activated_acl = to_acl_tensor(p->activated);
    workspace_size = 0;
    executor = nullptr;
    check_aclnn(
        aclnnSwiGluGetWorkspaceSize(
            gate_up_acl, -1, activated_acl, &workspace_size,
            &executor),
        "aclnnSwiGluGetWorkspaceSize");
    run_executor(
        "moe_fused_ascend", workspace_size, executor,
        &aclnnSwiGlu, stream);

    aclTensor *w2_acl = to_acl_tensor(p->w2);
    aclTensor *expert_output_acl = to_acl_tensor(p->expert_output);
    aclTensor *gmm_activated_items[]{activated_acl};
    aclTensor *gmm_w2_items[]{w2_acl};
    aclTensor *gmm_output_items[]{expert_output_acl};
    aclTensorList *gmm_activated = aclCreateTensorList(gmm_activated_items, 1);
    aclTensorList *gmm_w2 = aclCreateTensorList(gmm_w2_items, 1);
    aclTensorList *gmm_output = aclCreateTensorList(gmm_output_items, 1);
    workspace_size = 0;
    executor = nullptr;
    check_aclnn(
        aclnnGroupedMatmulV4GetWorkspaceSize(
            gmm_activated, gmm_w2, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, expert_counts_acl, nullptr,
            nullptr, nullptr, 2, 0, 1, 0, gmm_output, nullptr,
            nullptr, &workspace_size, &executor),
        "aclnnGroupedMatmulV4GetWorkspaceSize(w2)");
    run_executor(
        "moe_fused_ascend", workspace_size, executor,
        &aclnnGroupedMatmulV4, stream);

    aclTensor *absolute_row_ids_acl = to_acl_tensor(p->absolute_row_ids);
    workspace_size = 0;
    executor = nullptr;
    check_aclnn(
        aclnnAbsGetWorkspaceSize(
            expanded_row_ids_acl, absolute_row_ids_acl,
            &workspace_size, &executor),
        "aclnnAbsGetWorkspaceSize(row ids)");
    run_executor(
        "moe_fused_ascend", workspace_size, executor,
        &aclnnAbs, stream);

    int32_t local_expert_start = static_cast<int32_t>(p->local_expert_start);
    int32_t local_expert_end = static_cast<int32_t>(
        p->local_expert_start + p->local_num_experts);
    aclScalar *local_start_acl = aclCreateScalar(&local_expert_start, ACL_INT32);
    aclScalar *local_end_acl = aclCreateScalar(&local_expert_end, ACL_INT32);
    aclTensor *local_lower_mask_acl = to_acl_tensor(p->local_lower_mask);
    aclTensor *local_upper_mask_acl = to_acl_tensor(p->local_upper_mask);
    aclTensor *local_mask_acl = to_acl_tensor(p->local_mask);
    workspace_size = 0;
    executor = nullptr;
    check_aclnn(
        aclnnGeScalarGetWorkspaceSize(
            topk_ids_acl, local_start_acl, local_lower_mask_acl,
            &workspace_size, &executor),
        "aclnnGeScalarGetWorkspaceSize(local lower mask)");
    run_executor(
        "moe_fused_ascend", workspace_size, executor,
        &aclnnGeScalar, stream);
    workspace_size = 0;
    executor = nullptr;
    check_aclnn(
        aclnnLtScalarGetWorkspaceSize(
            topk_ids_acl, local_end_acl, local_upper_mask_acl,
            &workspace_size, &executor),
        "aclnnLtScalarGetWorkspaceSize(local upper mask)");
    run_executor(
        "moe_fused_ascend", workspace_size, executor,
        &aclnnLtScalar, stream);
    workspace_size = 0;
    executor = nullptr;
    check_aclnn(
        aclnnLogicalAndGetWorkspaceSize(
            local_lower_mask_acl, local_upper_mask_acl, local_mask_acl,
            &workspace_size, &executor),
        "aclnnLogicalAndGetWorkspaceSize(local mask)");
    run_executor(
        "moe_fused_ascend", workspace_size, executor,
        &aclnnLogicalAnd, stream);

    aclTensor *topk_weights_acl = to_acl_tensor(p->topk_weights);
    aclTensor *masked_weights_acl = to_acl_tensor(p->masked_weights);
    workspace_size = 0;
    executor = nullptr;
    check_aclnn(
        aclnnMulGetWorkspaceSize(
            topk_weights_acl, local_mask_acl, masked_weights_acl,
            &workspace_size, &executor),
        "aclnnMulGetWorkspaceSize(mask probabilities)");
    run_executor(
        "moe_fused_ascend", workspace_size, executor,
        &aclnnMul, stream);

    aclTensor *output_acl = to_acl_tensor(p->output);
    workspace_size = 0;
    executor = nullptr;
    check_aclnn(
        aclnnMoeTokenUnpermuteGetWorkspaceSize(
            expert_output_acl, absolute_row_ids_acl,
            masked_weights_acl, false, nullptr, output_acl,
            &workspace_size, &executor),
        "aclnnMoeTokenUnpermuteGetWorkspaceSize");
    run_executor(
        "moe_fused_ascend", workspace_size, executor,
        &aclnnMoeTokenUnpermute, stream);

    aclDestroyTensorList(gmm_x);
    aclDestroyTensorList(gmm_w13);
    aclDestroyTensorList(gmm_gate_up);
    aclDestroyTensorList(gmm_activated);
    aclDestroyTensorList(gmm_w2);
    aclDestroyTensorList(gmm_output);
    aclDestroyTensor(hidden_acl);
    aclDestroyTensor(topk_ids_acl);
    aclDestroyTensor(expanded_row_ids_acl);
    aclDestroyTensor(expert_counts_acl);
    aclDestroyTensor(expanded_scale_acl);
    aclDestroyTensor(local_lower_mask_acl);
    aclDestroyTensor(local_upper_mask_acl);
    aclDestroyTensor(local_mask_acl);
    aclDestroyScalar(local_start_acl);
    aclDestroyScalar(local_end_acl);
    aclDestroyTensor(topk_weights_acl);
    aclDestroyTensor(masked_weights_acl);
    aclDestroyTensor(absolute_row_ids_acl);
    aclDestroyTensor(output_acl);
    aclDestroyIntArray(range_acl);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MoeFusedDenseAscend::plan_dispatcher().registerDevice(
        Device::Type::ASCEND, &plan);
    MoeFusedDenseAscend::run_dispatcher().registerDevice(
        Device::Type::ASCEND, &run);
    MoeFusedDenseAscend::cleanup_dispatcher().registerDevice(
        Device::Type::ASCEND, &cleanup);
    return true;
}();

} // namespace infinicore::op::moe_fused_dense_ascend_impl::aclnn

#endif
