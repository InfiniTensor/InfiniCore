#ifdef ENABLE_ASCEND_API

#include "infinicore/ops/causal_conv1d_ascend_vendor.hpp"
#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(CausalConv1dAscendVendor);

namespace causal_conv1d_ascend_vendor_impl {

using GetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    const aclIntArray *,
    const aclIntArray *,
    const aclIntArray *,
    const aclIntArray *,
    int64_t,
    int64_t,
    int64_t,
    const aclTensor *,
    uint64_t *,
    aclOpExecutor **);

using RunFn = aclnnStatus (*)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

struct VendorApi {
    void *handle = nullptr;
    GetWorkspaceSizeFn get_workspace_size = nullptr;
    RunFn run = nullptr;
    std::string path;
};

static void configure_custom_opp_path() {
    const char *opp_override = std::getenv("INFINICORE_ASCEND_CUSTOM_OPP_PATH");
    const std::string opp_path = opp_override != nullptr
                                   ? opp_override
                                   : "/vllm-workspace/vllm-ascend/vllm_ascend/"
                                     "_cann_ops_custom/vendors/vllm-ascend";
    const char *current = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    const std::string current_path = current != nullptr ? current : "";
    if (current_path.find(opp_path) == std::string::npos) {
        const std::string combined = current_path.empty()
                                       ? opp_path
                                       : opp_path + ":" + current_path;
        setenv("ASCEND_CUSTOM_OPP_PATH", combined.c_str(), 1);
    }
}

static VendorApi load_vendor_api() {
    configure_custom_opp_path();

    std::vector<std::string> candidates;
    if (const char *override_path = std::getenv("INFINICORE_ASCEND_CAUSAL_CONV_VENDOR_SO")) {
        candidates.emplace_back(override_path);
    }
    candidates.emplace_back(
        "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/"
        "vendors/vllm-ascend/op_api/lib/libcust_opapi.so");
    candidates.emplace_back(
        "/usr/local/lib/python3.11/site-packages/vllm_ascend/"
        "_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so");

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

        auto get_workspace_size = reinterpret_cast<GetWorkspaceSizeFn>(
            dlsym(handle, "aclnnCausalConv1dGetWorkspaceSize"));
        auto run = reinterpret_cast<RunFn>(dlsym(handle, "aclnnCausalConv1d"));
        if (get_workspace_size != nullptr && run != nullptr) {
            return {handle, get_workspace_size, run, path};
        }

        errors += "\n  " + path + ": required symbols are missing";
        dlclose(handle);
    }

    throw std::runtime_error(
        "[causal_conv1d/ascend/vendor] unable to load vLLM-Ascend "
        "libcust_opapi.so. Set INFINICORE_ASCEND_CAUSAL_CONV_VENDOR_SO "
        "and INFINICORE_ASCEND_CUSTOM_OPP_PATH if installed elsewhere."
        + errors);
}

static VendorApi &vendor_api() {
    static VendorApi api = load_vendor_api();
    return api;
}

static aclDataType to_acl_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::F16:
        return ACL_FLOAT16;
    case DataType::BF16:
        return ACL_BF16;
    case DataType::F32:
        return ACL_FLOAT;
    default:
        throw std::runtime_error(
            "[causal_conv1d/ascend/vendor] only fp16, bf16 and fp32 "
            "are supported");
    }
}

static aclTensor *make_contiguous_acl_tensor(
    const Tensor &tensor,
    const std::vector<int64_t> &dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return aclCreateTensor(
        dims.data(),
        dims.size(),
        to_acl_dtype(tensor->dtype()),
        strides.data(),
        0,
        ACL_FORMAT_ND,
        dims.data(),
        dims.size(),
        const_cast<void *>(
            reinterpret_cast<const void *>(tensor->data())));
}

struct StreamWorkspace {
    void *ptr = nullptr;
    uint64_t capacity = 0;
    std::vector<void *> retired;
};

static void *acquire_stream_workspace(
    aclrtStream stream, uint64_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    static auto *mutex = new std::mutex();
    static auto *workspaces = new std::unordered_map<aclrtStream, StreamWorkspace>();
    std::lock_guard<std::mutex> lock(*mutex);
    auto &workspace = (*workspaces)[stream];
    if (workspace.capacity < bytes) {
        void *new_ptr = nullptr;
        auto ret = aclrtMalloc(&new_ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(
                "[causal_conv1d/ascend/vendor] workspace allocation "
                "failed: "
                + std::to_string(ret));
        }
        if (workspace.ptr != nullptr) {
            workspace.retired.push_back(workspace.ptr);
        }
        workspace.ptr = new_ptr;
        workspace.capacity = bytes;
    }
    return workspace.ptr;
}

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor state;
    graph::GraphTensor x;
    graph::GraphTensor weight;
    std::optional<graph::GraphTensor> bias;
    std::vector<int64_t> query_start_loc;
    std::vector<int64_t> cache_indices;
    bool fuse_silu;
    bool decode;
};

void *plan(
    Tensor out,
    Tensor state,
    const Tensor &x,
    const Tensor &weight,
    std::optional<Tensor> bias,
    std::vector<int64_t> query_start_loc,
    std::vector<int64_t> cache_indices,
    bool fuse_silu,
    bool decode) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(state),
        graph::GraphTensor(x),
        graph::GraphTensor(weight),
        bias.has_value()
            ? std::optional<graph::GraphTensor>(
                graph::GraphTensor(bias.value()))
            : std::nullopt,
        std::move(query_start_loc),
        std::move(cache_indices),
        fuse_silu,
        decode};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    infinicore::context::setDevice(p->x->device());

    const auto &x_shape = p->x->shape();
    const auto &weight_shape = p->weight->shape();
    const auto &state_shape = p->state->shape();
    if (x_shape.size() != 3 || x_shape[0] != 1
        || weight_shape.size() != 2 || state_shape.size() != 3) {
        throw std::runtime_error(
            "[causal_conv1d/ascend/vendor] expected x [1,T,C], "
            "weight [K,C], state [pool,K-1,C]");
    }
    const int64_t tokens = static_cast<int64_t>(x_shape[1]);
    const int64_t channels = static_cast<int64_t>(x_shape[2]);
    const int64_t kernel = static_cast<int64_t>(weight_shape[0]);
    if (weight_shape[1] != x_shape[2]
        || state_shape[1] + 1 != weight_shape[0]
        || state_shape[2] != x_shape[2]
        || p->query_start_loc.size() != p->cache_indices.size() + 1) {
        throw std::runtime_error(
            "[causal_conv1d/ascend/vendor] incompatible shapes or "
            "metadata lengths");
    }
    if (!p->x->is_contiguous() || !p->out->is_contiguous()
        || !p->weight->is_contiguous() || !p->state->is_contiguous()) {
        throw std::runtime_error(
            "[causal_conv1d/ascend/vendor] all tensors must be contiguous");
    }

    Tensor x(p->x);
    Tensor out(p->out);
    Tensor weight(p->weight);
    Tensor state(p->state);
    auto *x_acl = make_contiguous_acl_tensor(x, {tokens, channels});
    auto *weight_acl = make_contiguous_acl_tensor(weight, {kernel, channels});
    auto *state_acl = make_contiguous_acl_tensor(
        state,
        {static_cast<int64_t>(state_shape[0]),
         static_cast<int64_t>(state_shape[1]),
         channels});
    auto *out_acl = make_contiguous_acl_tensor(out, {tokens, channels});
    aclTensor *bias_acl = nullptr;
    if (p->bias.has_value()) {
        Tensor bias(p->bias.value());
        bias_acl = make_contiguous_acl_tensor(bias, {channels});
    }

    std::vector<int64_t> initial_state_mode(
        p->cache_indices.size(), p->decode ? 1 : 0);
    aclIntArray *query_start_loc_acl = aclCreateIntArray(
        p->query_start_loc.data(), p->query_start_loc.size());
    aclIntArray *cache_indices_acl = aclCreateIntArray(
        p->cache_indices.data(), p->cache_indices.size());
    aclIntArray *initial_state_mode_acl = aclCreateIntArray(
        initial_state_mode.data(), initial_state_mode.size());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto &api = vendor_api();
    auto ret = api.get_workspace_size(
        x_acl,
        weight_acl,
        bias_acl,
        state_acl,
        query_start_loc_acl,
        cache_indices_acl,
        initial_state_mode_acl,
        nullptr,
        p->fuse_silu ? 1 : 0,
        0,
        p->decode ? 1 : 0,
        out_acl,
        &workspace_size,
        &executor);
    if (ret != 0) {
        const char *message = aclGetRecentErrMsg();
        throw std::runtime_error(
            "[causal_conv1d/ascend/vendor] "
            "aclnnCausalConv1dGetWorkspaceSize failed: "
            + std::to_string(ret) + ", "
            + (message != nullptr ? message : "(no ACL error)"));
    }

    auto stream = reinterpret_cast<aclrtStream>(infinicore::context::getStream());
    void *workspace = acquire_stream_workspace(stream, workspace_size);
    ret = api.run(workspace, workspace_size, executor, stream);

    aclDestroyTensor(x_acl);
    aclDestroyTensor(weight_acl);
    aclDestroyTensor(state_acl);
    aclDestroyTensor(out_acl);
    if (bias_acl != nullptr) {
        aclDestroyTensor(bias_acl);
    }
    aclDestroyIntArray(query_start_loc_acl);
    aclDestroyIntArray(cache_indices_acl);
    aclDestroyIntArray(initial_state_mode_acl);

    if (ret != 0) {
        const char *message = aclGetRecentErrMsg();
        throw std::runtime_error(
            "[causal_conv1d/ascend/vendor] aclnnCausalConv1d failed: "
            + std::to_string(ret) + ", "
            + (message != nullptr ? message : "(no ACL error)"));
    }
}

void cleanup(void **planned_meta_ptr) {
    auto *p = *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    delete p;
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    // This must happen while libinfinicore is loaded, before the first ACLNN
    // operator initializes the custom OPP registry.
    configure_custom_opp_path();
    CausalConv1dAscendVendor::plan_dispatcher().registerDevice(
        Device::Type::ASCEND, &plan);
    CausalConv1dAscendVendor::run_dispatcher().registerDevice(
        Device::Type::ASCEND, &run);
    CausalConv1dAscendVendor::cleanup_dispatcher().registerDevice(
        Device::Type::ASCEND, &cleanup);
    return true;
}();

} // namespace causal_conv1d_ascend_vendor_impl

CausalConv1dAscendVendor::CausalConv1dAscendVendor(
    Tensor out,
    Tensor conv_state,
    const Tensor &x,
    const Tensor &weight,
    std::optional<Tensor> bias,
    std::vector<int64_t> query_start_loc,
    std::vector<int64_t> cache_indices,
    bool fuse_silu,
    bool decode) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, conv_state, x, weight);
    INFINICORE_GRAPH_OP_DISPATCH(
        out->device().getType(),
        out,
        conv_state,
        x,
        weight,
        bias,
        std::move(query_start_loc),
        std::move(cache_indices),
        fuse_silu,
        decode);
}

void CausalConv1dAscendVendor::execute(
    Tensor out,
    Tensor conv_state,
    const Tensor &x,
    const Tensor &weight,
    std::optional<Tensor> bias,
    std::vector<int64_t> query_start_loc,
    std::vector<int64_t> cache_indices,
    bool fuse_silu,
    bool decode) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        CausalConv1dAscendVendor,
        out,
        conv_state,
        x,
        weight,
        bias,
        std::move(query_start_loc),
        std::move(cache_indices),
        fuse_silu,
        decode);
}

Tensor causal_conv1d_ascend_vendor(
    const Tensor &x,
    Tensor conv_state,
    const Tensor &weight,
    std::optional<Tensor> bias,
    std::vector<int64_t> query_start_loc,
    std::vector<int64_t> cache_indices,
    bool fuse_silu,
    bool decode) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    CausalConv1dAscendVendor::execute(
        out,
        conv_state,
        x,
        weight,
        bias,
        std::move(query_start_loc),
        std::move(cache_indices),
        fuse_silu,
        decode);
    return out;
}

} // namespace infinicore::op

#endif
