#ifdef ENABLE_ASCEND_API

#include "infinicore/ops/matmul_allreduce_add_rmsnorm_ascend.hpp"
#include "../../../infiniccl/infiniccl_impl.h"
#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" int HcclGetCommName(
    void *communicator, char *communicator_name);

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(
    MatmulAllReduceAddRmsNormAscend);

namespace matmul_allreduce_add_rmsnorm_ascend_impl {

using GetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    char *,
    int64_t,
    int64_t,
    double,
    bool,
    bool,
    const aclTensor *,
    const aclTensor *,
    uint64_t *,
    aclOpExecutor **);

using RunFn = aclnnStatus (*)(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

using AddRmsNormGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    double,
    const aclTensor *,
    const aclTensor *,
    const aclTensor *,
    uint64_t *,
    aclOpExecutor **);

struct VendorApi {
    void *handle = nullptr;
    GetWorkspaceSizeFn get_workspace_size = nullptr;
    RunFn run = nullptr;
    AddRmsNormGetWorkspaceSizeFn add_rmsnorm_get_workspace_size = nullptr;
    RunFn add_rmsnorm_run = nullptr;
    std::string path;
};

static void configure_custom_opp_path() {
    const char *override_path = std::getenv("INFINICORE_ASCEND_CUSTOM_OPP_PATH");
    const std::string opp_path = override_path != nullptr
                                   ? override_path
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
    if (const char *override_path = std::getenv(
            "INFINICORE_ASCEND_MC2_ADD_RMSNORM_VENDOR_SO")) {
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
                    + (error != nullptr ? error
                                        : "unknown dlopen error");
            continue;
        }
        auto get_workspace_size = reinterpret_cast<GetWorkspaceSizeFn>(dlsym(
            handle,
            "aclnnMatmulAllreduceAddRmsnormGetWorkspaceSize"));
        auto run = reinterpret_cast<RunFn>(dlsym(
            handle, "aclnnMatmulAllreduceAddRmsnorm"));
        auto add_rmsnorm_get_workspace_size = reinterpret_cast<AddRmsNormGetWorkspaceSizeFn>(dlsym(
            handle, "aclnnAddRmsNormBiasGetWorkspaceSize"));
        auto add_rmsnorm_run = reinterpret_cast<RunFn>(dlsym(
            handle, "aclnnAddRmsNormBias"));
        if (get_workspace_size != nullptr && run != nullptr
            && add_rmsnorm_get_workspace_size != nullptr
            && add_rmsnorm_run != nullptr) {
            return {handle, get_workspace_size, run,
                    add_rmsnorm_get_workspace_size,
                    add_rmsnorm_run, path};
        }
        errors += "\n  " + path + ": required symbols are missing";
        dlclose(handle);
    }
    throw std::runtime_error(
        "[matmul_allreduce_add_rmsnorm/ascend/vendor] unable to "
        "load vLLM-Ascend libcust_opapi.so. Set "
        "INFINICORE_ASCEND_MC2_ADD_RMSNORM_VENDOR_SO and "
        "INFINICORE_ASCEND_CUSTOM_OPP_PATH if installed elsewhere."
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
    default:
        throw std::runtime_error(
            "[matmul_allreduce_add_rmsnorm/ascend/vendor] only "
            "fp16 and bf16 are supported");
    }
}

static aclTensor *make_acl_tensor(const Tensor &tensor) {
    const auto &shape = tensor->shape();
    const auto &strides = tensor->strides();
    std::vector<int64_t> dims(shape.begin(), shape.end());
    std::vector<int64_t> acl_strides(
        strides.begin(), strides.end());
    return aclCreateTensor(
        dims.data(),
        dims.size(),
        to_acl_dtype(tensor->dtype()),
        acl_strides.data(),
        0,
        ACL_FORMAT_ND,
        dims.data(),
        dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(
            tensor->data())));
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
    thread_local std::unordered_map<
        aclrtStream, StreamWorkspace>
        workspaces;
    auto &workspace = workspaces[stream];
    if (workspace.capacity < bytes) {
        void *new_ptr = nullptr;
        auto ret = aclrtMalloc(
            &new_ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(
                "[matmul_allreduce_add_rmsnorm/ascend/vendor] "
                "workspace allocation failed: "
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

static void *acquire_stream_rstd_buffer(
    aclrtStream stream, uint64_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    // The vendor op writes rstd asynchronously. Keep a permanent buffer per
    // stream instead of returning a temporary Tensor to the allocator before
    // the stream has consumed it.
    thread_local std::unordered_map<
        aclrtStream, StreamWorkspace>
        buffers;
    auto &buffer = buffers[stream];
    if (buffer.capacity < bytes) {
        void *new_ptr = nullptr;
        auto ret = aclrtMalloc(
            &new_ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(
                "[add_rmsnorm/ascend/vendor] rstd buffer "
                "allocation failed: "
                + std::to_string(ret));
        }
        if (buffer.ptr != nullptr) {
            // Do not free an older generation while work on this stream may
            // still reference it. Shape growth is rare and bounded.
            buffer.retired.push_back(buffer.ptr);
        }
        buffer.ptr = new_ptr;
        buffer.capacity = bytes;
    }
    return buffer.ptr;
}

static const std::string &get_group_name(
    infinicclComm_t communicator) {
    if (communicator == nullptr || communicator->comm == nullptr) {
        throw std::runtime_error(
            "[matmul_allreduce_add_rmsnorm/ascend/vendor] "
            "communicator is null");
    }
    thread_local std::unordered_map<
        infinicclComm_t, std::string>
        names;
    auto found = names.find(communicator);
    if (found != names.end()) {
        return found->second;
    }
    char name[128] = {};
    auto ret = HcclGetCommName(communicator->comm, name);
    if (ret != 0) {
        throw std::runtime_error(
            "[matmul_allreduce_add_rmsnorm/ascend/vendor] "
            "HcclGetCommName failed: "
            + std::to_string(ret));
    }
    return names.emplace(communicator, name).first->second;
}

struct PlannedMeta {
    graph::GraphTensor normalized;
    graph::GraphTensor add_out;
    graph::GraphTensor input;
    graph::GraphTensor weight;
    graph::GraphTensor residual;
    graph::GraphTensor gamma;
    infinicclComm_t communicator;
    float epsilon;
};

void *plan(
    Tensor normalized,
    Tensor add_out,
    const Tensor &input,
    const Tensor &weight,
    const Tensor &residual,
    const Tensor &gamma,
    infinicclComm_t communicator,
    float epsilon) {
    return new PlannedMeta{
        graph::GraphTensor(normalized),
        graph::GraphTensor(add_out),
        graph::GraphTensor(input),
        graph::GraphTensor(weight),
        graph::GraphTensor(residual),
        graph::GraphTensor(gamma),
        communicator,
        epsilon};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    infinicore::context::setDevice(p->input->device());

    if (p->input->ndim() != 2 || p->weight->ndim() != 2
        || p->residual->ndim() != 2 || p->gamma->ndim() != 1
        || p->normalized->ndim() != 2
        || p->add_out->ndim() != 2) {
        throw std::runtime_error(
            "[matmul_allreduce_add_rmsnorm/ascend/vendor] "
            "expected 2D matrices and 1D gamma");
    }
    const auto rows = p->input->shape()[0];
    const auto out_features = p->weight->shape()[0];
    if (p->input->shape()[1] != p->weight->shape()[1]
        || p->residual->shape()[0] != rows
        || p->residual->shape()[1] != out_features
        || p->gamma->shape()[0] != out_features
        || p->normalized->shape() != p->residual->shape()
        || p->add_out->shape() != p->residual->shape()) {
        throw std::runtime_error(
            "[matmul_allreduce_add_rmsnorm/ascend/vendor] "
            "incompatible tensor shapes");
    }
    if (!p->input->is_contiguous() || !p->weight->is_contiguous()
        || !p->residual->is_contiguous()
        || !p->gamma->is_contiguous()
        || !p->normalized->is_contiguous()
        || !p->add_out->is_contiguous()) {
        throw std::runtime_error(
            "[matmul_allreduce_add_rmsnorm/ascend/vendor] all "
            "tensors must be contiguous");
    }

    Tensor input(p->input);
    Tensor weight(p->weight);
    Tensor residual(p->residual);
    Tensor gamma(p->gamma);
    Tensor normalized(p->normalized);
    Tensor add_out(p->add_out);
    aclTensor *input_acl = make_acl_tensor(input);
    aclTensor *weight_acl = make_acl_tensor(weight);
    aclTensor *residual_acl = make_acl_tensor(residual);
    aclTensor *gamma_acl = make_acl_tensor(gamma);
    aclTensor *normalized_acl = make_acl_tensor(normalized);
    aclTensor *add_out_acl = make_acl_tensor(add_out);
    std::vector<aclTensor *> tensors{
        input_acl, weight_acl, residual_acl, gamma_acl,
        normalized_acl, add_out_acl};
    for (auto *tensor : tensors) {
        if (tensor == nullptr) {
            for (auto *created : tensors) {
                if (created != nullptr) {
                    aclDestroyTensor(created);
                }
            }
            throw std::runtime_error(
                "[matmul_allreduce_add_rmsnorm/ascend/vendor] "
                "aclCreateTensor failed");
        }
    }

    const std::string &group = get_group_name(p->communicator);
    std::vector<char> mutable_group(group.begin(), group.end());
    mutable_group.push_back('\0');
    auto &api = vendor_api();
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = api.get_workspace_size(
        input_acl,
        weight_acl,
        residual_acl,
        gamma_acl,
        mutable_group.data(),
        p->communicator->world_size,
        p->communicator->rank,
        static_cast<double>(p->epsilon),
        true,
        false,
        normalized_acl,
        add_out_acl,
        &workspace_size,
        &executor);
    if (ret == ACL_SUCCESS) {
        auto stream = reinterpret_cast<aclrtStream>(
            infinicore::context::getStream());
        void *workspace = acquire_stream_workspace(stream, workspace_size);
        ret = api.run(
            workspace, workspace_size, executor, stream);
    }

    for (auto *tensor : tensors) {
        aclDestroyTensor(tensor);
    }
    if (ret != ACL_SUCCESS) {
        const char *message = aclGetRecentErrMsg();
        throw std::runtime_error(
            "[matmul_allreduce_add_rmsnorm/ascend/vendor] call "
            "failed: "
            + std::to_string(ret) + ", "
            + (message != nullptr ? message : "(no ACL error)"));
    }
}

void cleanup(void **planned_meta_ptr) {
    auto *p = *reinterpret_cast<PlannedMeta **>(
        planned_meta_ptr);
    delete p;
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MatmulAllReduceAddRmsNormAscend::plan_dispatcher()
        .registerDevice(Device::Type::ASCEND, &plan);
    MatmulAllReduceAddRmsNormAscend::run_dispatcher()
        .registerDevice(Device::Type::ASCEND, &run);
    MatmulAllReduceAddRmsNormAscend::cleanup_dispatcher()
        .registerDevice(Device::Type::ASCEND, &cleanup);
    return true;
}();

} // namespace matmul_allreduce_add_rmsnorm_ascend_impl

MatmulAllReduceAddRmsNormAscend::
    MatmulAllReduceAddRmsNormAscend(
        Tensor normalized,
        Tensor add_out,
        const Tensor &input,
        const Tensor &weight,
        const Tensor &residual,
        const Tensor &gamma,
        infinicclComm_t communicator,
        float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        normalized, add_out, input, weight, residual, gamma);
    INFINICORE_GRAPH_OP_DISPATCH(
        normalized->device().getType(),
        normalized,
        add_out,
        input,
        weight,
        residual,
        gamma,
        communicator,
        epsilon);
}

void MatmulAllReduceAddRmsNormAscend::execute(
    Tensor normalized,
    Tensor add_out,
    const Tensor &input,
    const Tensor &weight,
    const Tensor &residual,
    const Tensor &gamma,
    infinicclComm_t communicator,
    float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MatmulAllReduceAddRmsNormAscend,
        normalized,
        add_out,
        input,
        weight,
        residual,
        gamma,
        communicator,
        epsilon);
}

std::tuple<Tensor, Tensor>
matmul_allreduce_add_rmsnorm_ascend(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &residual,
    const Tensor &gamma,
    infinicclComm_t communicator,
    float epsilon) {
    auto normalized = Tensor::empty(
        residual->shape(), residual->dtype(), residual->device());
    auto add_out = Tensor::empty(
        residual->shape(), residual->dtype(), residual->device());
    MatmulAllReduceAddRmsNormAscend::execute(
        normalized,
        add_out,
        input,
        weight,
        residual,
        gamma,
        communicator,
        epsilon);
    return {normalized, add_out};
}

std::tuple<Tensor, Tensor>
add_rmsnorm_ascend_vendor(
    const Tensor &x1,
    const Tensor &x2,
    const Tensor &gamma,
    float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x1, x2, gamma);
    infinicore::context::setDevice(x1->device());
    if (x1->shape() != x2->shape() || x1->ndim() < 2
        || gamma->ndim() != 1
        || gamma->shape()[0] != x1->shape().back()) {
        throw std::runtime_error(
            "[add_rmsnorm/ascend/vendor] incompatible tensor shapes");
    }
    if (x1->dtype() != x2->dtype()
        || x1->dtype() != gamma->dtype()) {
        throw std::runtime_error(
            "[add_rmsnorm/ascend/vendor] tensor dtypes must match");
    }
    if (!x1->is_contiguous() || !x2->is_contiguous()
        || !gamma->is_contiguous()) {
        throw std::runtime_error(
            "[add_rmsnorm/ascend/vendor] all tensors must be contiguous");
    }

    auto normalized = Tensor::empty(
        x1->shape(), x1->dtype(), x1->device());
    auto add_out = Tensor::empty(
        x1->shape(), x1->dtype(), x1->device());
    aclTensor *x1_acl = matmul_allreduce_add_rmsnorm_ascend_impl::make_acl_tensor(x1);
    aclTensor *x2_acl = matmul_allreduce_add_rmsnorm_ascend_impl::make_acl_tensor(x2);
    aclTensor *gamma_acl = matmul_allreduce_add_rmsnorm_ascend_impl::make_acl_tensor(gamma);
    aclTensor *normalized_acl = matmul_allreduce_add_rmsnorm_ascend_impl::make_acl_tensor(normalized);
    aclTensor *add_out_acl = matmul_allreduce_add_rmsnorm_ascend_impl::make_acl_tensor(add_out);

    std::vector<int64_t> rstd_dims;
    rstd_dims.reserve(x1->ndim());
    for (size_t i = 0; i < x1->ndim(); ++i) {
        rstd_dims.push_back(
            static_cast<int64_t>(x1->shape()[i]));
    }
    rstd_dims.back() = 1;
    std::vector<int64_t> rstd_strides(rstd_dims.size(), 1);
    for (ptrdiff_t i = static_cast<ptrdiff_t>(rstd_dims.size()) - 2;
         i >= 0; --i) {
        rstd_strides[i] = rstd_strides[i + 1] * rstd_dims[i + 1];
    }
    uint64_t rstd_numel = 1;
    for (auto dim : rstd_dims) {
        rstd_numel *= static_cast<uint64_t>(dim);
    }
    auto stream = reinterpret_cast<aclrtStream>(
        infinicore::context::getStream());
    void *rstd_ptr = matmul_allreduce_add_rmsnorm_ascend_impl::acquire_stream_rstd_buffer(
        stream, rstd_numel * sizeof(float));
    aclTensor *rstd_acl = aclCreateTensor(
        rstd_dims.data(),
        rstd_dims.size(),
        ACL_FLOAT,
        rstd_strides.data(),
        0,
        ACL_FORMAT_ND,
        rstd_dims.data(),
        rstd_dims.size(),
        rstd_ptr);

    std::vector<aclTensor *> tensors{
        x1_acl, x2_acl, gamma_acl, normalized_acl,
        rstd_acl, add_out_acl};
    for (auto *tensor : tensors) {
        if (tensor == nullptr) {
            for (auto *created : tensors) {
                if (created != nullptr) {
                    aclDestroyTensor(created);
                }
            }
            throw std::runtime_error(
                "[add_rmsnorm/ascend/vendor] aclCreateTensor failed");
        }
    }

    auto &api = matmul_allreduce_add_rmsnorm_ascend_impl::vendor_api();
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = api.add_rmsnorm_get_workspace_size(
        x1_acl,
        x2_acl,
        gamma_acl,
        nullptr,
        static_cast<double>(epsilon),
        normalized_acl,
        rstd_acl,
        add_out_acl,
        &workspace_size,
        &executor);
    if (ret == ACL_SUCCESS) {
        void *workspace = matmul_allreduce_add_rmsnorm_ascend_impl::
            acquire_stream_workspace(stream, workspace_size);
        ret = api.add_rmsnorm_run(
            workspace, workspace_size, executor, stream);
    }

    for (auto *tensor : tensors) {
        aclDestroyTensor(tensor);
    }
    if (ret != ACL_SUCCESS) {
        const char *message = aclGetRecentErrMsg();
        throw std::runtime_error(
            "[add_rmsnorm/ascend/vendor] call failed: "
            + std::to_string(ret) + ", "
            + (message != nullptr ? message : "(no ACL error)"));
    }
    return {normalized, add_out};
}

} // namespace infinicore::op

#endif
