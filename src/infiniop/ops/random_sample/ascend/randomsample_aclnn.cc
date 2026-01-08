#include "../../../devices/ascend/common_ascend.h"
#include "random_sample_aclnn.h"
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_topk.h>

namespace op::random_sample::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t probs;
    aclnnTensorDescriptor_t result;

    ~Opaque() {
        delete probs;
        delete result;
    }
};

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);
    CHECK_DTYPE(result->dt_i,
                INFINI_DTYPE_I64,
                INFINI_DTYPE_I32,
                INFINI_DTYPE_I16,
                INFINI_DTYPE_U64,
                INFINI_DTYPE_U32);

    auto workspace_size = probs_desc->numel() * infiniSizeOf(probs_desc->dtype())               // topk_val tensor
                        + probs_desc->numel() * infiniSizeOf(infiniDtype_t::INFINI_DTYPE_I64)   // topk_idx tensor
                        + result_desc->numel() * infiniSizeOf(infiniDtype_t::INFINI_DTYPE_I64); // sample tensor

    auto tresult = new aclnnTensorDescriptor(result_desc);
    auto tprobs = new aclnnTensorDescriptor(probs_desc);
    *desc_ptr
        = new Descriptor(
            result.take(),
            workspace_size,
            new Opaque{tprobs, tresult},
            handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

extern "C" infiniStatus_t random_sample_kernel_launch(
    void *probs,
    void *result,
    void *topk_val_addr,
    void *topk_idx_addr,
    float random_val,
    float topp,
    int topk,
    float temperature,
    uint64_t n,
    infiniDtype_t dt_p,
    void *stream);

infiniStatus_t
Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const {
    if (workspace_size < _min_workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto topk_ = topk <= (int)_info.n ? topk : (int)_info.n;
    bool dosample = topk_ > 1 && temperature != 0.0f && topp != 0.0f && random_val != 0.0f;
    auto topk_shape = std::vector<int64_t>{dosample ? topk_ : 1};
    auto topk_stride = std::vector<int64_t>{1};
    // AclnnTopk only supports int64 index output
    auto topk_idx = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_I64), topk_shape, topk_stride);
    auto topk_val = new aclnnTensorDescriptor(toAclDataType(_info.dt_p), topk_shape, topk_stride);

    auto topk_val_addr = workspace;
    auto topk_idx_addr = (void *)((uint8_t *)workspace + topk_ * infiniSizeOf(_info.dt_p));
    auto sample_addr = (void *)((uint8_t *)topk_idx_addr + topk_ * infiniSizeOf(INFINI_DTYPE_I64));

    // Step 1: TopK, Aclnn Topk only supports int64 index output
    uint64_t topk_workspace_size = 0;
    aclOpExecutor *topk_executor = nullptr;
    CHECK_ACL(aclnnTopkGetWorkspaceSize(_opaque->probs->tensor,
                                        topk_shape[0],
                                        0,
                                        true,
                                        true,
                                        topk_val->tensor,
                                        topk_idx->tensor,
                                        &topk_workspace_size,
                                        &topk_executor));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(topk_executor));
    void *topk_workspace;
    CHECK_ACL(aclrtMalloc(&topk_workspace, topk_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    AclSetTensorAddr(topk_executor, 0, _opaque->probs->tensor, (void *)probs);
    AclSetTensorAddr(topk_executor, 1, topk_val->tensor, topk_val_addr);
    AclSetTensorAddr(topk_executor, 2, topk_idx->tensor, topk_idx_addr);

    CHECK_ACL(aclnnTopk(topk_workspace, topk_workspace_size, topk_executor, stream));
    CHECK_ACL(aclrtFree(topk_workspace));
    // Step 2: Do random sample if needed
    if (dosample) {
        auto status = random_sample_kernel_launch((void *)probs, sample_addr, topk_val_addr, topk_idx_addr, random_val, topp, topk_, temperature, _info.n, _info.dt_p, stream);
        CHECK_STATUS(status);
    } else {
        // If not sampling, just copy the topk_idx to sample_addr
        CHECK_ACL(aclrtMemcpy(sample_addr, infiniSizeOf(INFINI_DTYPE_I64) * 1, topk_idx_addr, infiniSizeOf(INFINI_DTYPE_I64) * 1, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    // Step 3: Cast to result dtype if needed
    if (_info.dt_i != INFINI_DTYPE_I64) {
        auto cast_input = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_I64), {}, {});
        uint64_t cast_workspace_size = 0;
        aclOpExecutor *cast_executor = nullptr;
        CHECK_ACL(aclnnCastGetWorkspaceSize(cast_input->tensor, toAclDataType(_info.dt_i), _opaque->result->tensor, &cast_workspace_size, &cast_executor));
        CHECK_ACL(aclSetAclOpExecutorRepeatable(cast_executor));
        AclSetTensorAddr(cast_executor, 0, cast_input->tensor, sample_addr);
        AclSetTensorAddr(cast_executor, 1, _opaque->result->tensor, result);
        CHECK_ACL(aclnnCast(nullptr, cast_workspace_size, cast_executor, stream));
    } else {
        // If no casting is needed, just copy the data
        CHECK_ACL(aclrtMemcpy(result, infiniSizeOf(_info.dt_i) * 1, sample_addr, infiniSizeOf(INFINI_DTYPE_I64) * 1, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::random_sample::ascend
