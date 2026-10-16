#if defined(ENABLE_ASCEND_API)

#include "../../../infiniccl/infiniccl_impl.h"
#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/linear_allreduce.hpp"

#include <acl/acl.h>
#include <aclnnop/aclnn_matmul_all_reduce.h>
#include <hccl/hccl.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace infinicore::op::linear_allreduce_impl::ascend {

// ---- workspace pool -----------------------------------------
// Per-stream reusable workspace.  Once grown to the maximum
// required size, the buffer is never freed (leaked on growth).
// This avoids the MC2-notify-after-free bug: MC2 callbacks may
// still reference old workspace after stream sync completes.
struct WorkspaceBuf {
    void *ptr = nullptr;
    size_t cap = 0;
};

class WorkspacePool {
    std::unordered_map<aclrtStream, WorkspaceBuf> bufs_;
    std::mutex mtx_;

public:
    void *ensure(aclrtStream stream, size_t need) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto &b = bufs_[stream];
        if (need <= b.cap) {
            return b.ptr;
        }
        void *new_ptr = nullptr;
        aclError rc = aclrtMalloc(&new_ptr, need, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != ACL_SUCCESS || !new_ptr) {
            fprintf(stderr, "[linear_allreduce/ascend] FATAL: aclrtMalloc(%zu MB) "
                            "failed rc=%d\n",
                    need / (1024 * 1024), (int)rc);
            fflush(stderr);
            throw std::runtime_error("[linear_allreduce/ascend] workspace alloc failed");
        }
        b.ptr = new_ptr;
        b.cap = need;
        return b.ptr;
    }
};

static WorkspacePool g_pool;

static inline HcclComm get_hccl_comm(infinicclComm_t comm) {
    return static_cast<HcclComm>(comm->comm);
}

static aclDataType to_acl_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::F16:
        return ACL_FLOAT16;
    case DataType::BF16:
        return ACL_BF16;
    default:
        throw std::runtime_error(
            "[linear_allreduce/ascend] unsupported dtype: " + std::to_string(static_cast<int>(dtype)) + ". aclnnMatmulAllReduce only supports F16/BF16");
    }
}

void linear_allreduce_impl(
    Tensor output, Tensor input, Tensor weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator) {
    infinicore::context::setDevice(input->device());

    auto atype = input->dtype();
    if (atype != DataType::F16 && atype != DataType::BF16) {
        throw std::runtime_error(
            "[linear_allreduce/ascend] unsupported activation dtype: " + std::to_string(static_cast<int>(atype)) + ". aclnnMatmulAllReduce only supports F16/BF16");
    }

    auto w_perm = weight->permute({1, 0});
    Tensor weight_w = w_perm->is_contiguous() ? Tensor(w_perm) : w_perm->contiguous();

    auto in_shape = input->shape();
    auto wt_shape = weight_w->shape();
    auto out_shape = output->shape();

    std::vector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    std::vector<int64_t> wt_dims(wt_shape.begin(), wt_shape.end());
    std::vector<int64_t> out_dims(out_shape.begin(), out_shape.end());

    aclTensor *x1_acl = aclCreateTensor(
        in_dims.data(), in_dims.size(),
        to_acl_dtype(atype),
        nullptr, 0, ACL_FORMAT_ND,
        in_dims.data(), in_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(input->data())));

    aclTensor *x2_acl = aclCreateTensor(
        wt_dims.data(), wt_dims.size(),
        to_acl_dtype(atype),
        nullptr, 0, ACL_FORMAT_ND,
        wt_dims.data(), wt_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(weight_w->data())));

    aclTensor *bias_acl = nullptr;
    if (bias.has_value()) {
        Tensor bias_w = bias.value()->is_contiguous() ? Tensor(bias.value())
                                                      : bias.value()->contiguous();
        auto bias_shape = bias_w->shape();
        std::vector<int64_t> bias_dims(bias_shape.begin(), bias_shape.end());
        bias_acl = aclCreateTensor(
            bias_dims.data(), bias_dims.size(),
            to_acl_dtype(atype),
            nullptr, 0, ACL_FORMAT_ND,
            bias_dims.data(), bias_dims.size(),
            const_cast<void *>(reinterpret_cast<const void *>(bias_w->data())));
    }

    aclTensor *out_acl = aclCreateTensor(
        out_dims.data(), out_dims.size(),
        to_acl_dtype(atype),
        nullptr, 0, ACL_FORMAT_ND,
        out_dims.data(), out_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(output->data())));

    HcclComm hccl_comm = get_hccl_comm(communicator);
    char group_name[COMM_NAME_MAX_LENGTH] = {};
    HcclGetCommName(hccl_comm, group_name);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnMatmulAllReduceGetWorkspaceSize(
        x1_acl, x2_acl, bias_acl,
        group_name, "sum",
        0, 1,
        out_acl,
        &workspace_size, &executor);

    if (ret != 0) {
        if (bias_acl) {
            aclDestroyTensor(bias_acl);
        }
        aclDestroyTensor(x1_acl);
        aclDestroyTensor(x2_acl);
        aclDestroyTensor(out_acl);
        const char *err = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string("[linear_allreduce/ascend] GetWorkspaceSize failed: ") + std::to_string(ret) + ", msg: " + (err ? err : "(null)"));
    }

    aclrtStream stream = static_cast<aclrtStream>(
        infinicore::context::getStream());
    void *workspace = g_pool.ensure(stream, (size_t)workspace_size);

    ret = aclnnMatmulAllReduce(
        workspace, workspace_size, executor, stream);

    if (bias_acl) {
        aclDestroyTensor(bias_acl);
    }
    aclDestroyTensor(x1_acl);
    aclDestroyTensor(x2_acl);
    aclDestroyTensor(out_acl);

    if (ret != 0) {
        const char *err = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string("[linear_allreduce/ascend] execution failed: ") + std::to_string(ret) + ", msg: " + (err ? err : "(null)"));
    }
}

} // namespace infinicore::op::linear_allreduce_impl::ascend

#endif // ENABLE_ASCEND_API
