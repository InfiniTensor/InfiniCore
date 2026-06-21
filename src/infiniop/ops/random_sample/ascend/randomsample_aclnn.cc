#include "../../../devices/ascend/common_ascend.h"
#include "random_sample_aclnn.h"
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
    CHECK_DTYPE(result->dt_i, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
    auto topk_val_dtype = (probs_desc->dtype() == INFINI_DTYPE_BF16 || probs_desc->dtype() == INFINI_DTYPE_F16)
                            ? INFINI_DTYPE_F32
                            : probs_desc->dtype();
    auto workspace_size = utils::align(probs_desc->numel() * infiniSizeOf(topk_val_dtype), 32)
                        + probs_desc->numel() * infiniSizeOf(INFINI_DTYPE_I64);
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
    infiniDtype_t dt_i,
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
    auto effective_topk = dosample ? topk_ : 1;
    auto topk_shape = std::vector<int64_t>{effective_topk};
    auto topk_stride = std::vector<int64_t>{1};

    bool use_fp32_topk = (_info.dt_p == INFINI_DTYPE_BF16 || _info.dt_p == INFINI_DTYPE_F16);

    void *probs_for_topk = const_cast<void *>(probs);
    void *topk_val_addr = workspace;
    auto topk_val_bytes = effective_topk * infiniSizeOf(use_fp32_topk ? INFINI_DTYPE_F32 : _info.dt_p);
    void *topk_idx_addr = (void *)((uint8_t *)topk_val_addr + utils::align(topk_val_bytes, 32));

    uint64_t topk_workspace_size = 0;
    aclOpExecutor *topk_executor = nullptr;

    if (use_fp32_topk) {
        void *probs_fp32;
        auto probs_fp32_size = _info.n * infiniSizeOf(INFINI_DTYPE_F32);
        CHECK_ACL(aclrtMalloc(&probs_fp32, probs_fp32_size, ACL_MEM_MALLOC_HUGE_FIRST));

        void *probs_host;
        auto probs_host_size = _info.n * infiniSizeOf(_info.dt_p);
        CHECK_ACL(aclrtMallocHost(&probs_host, probs_host_size));
        void *probs_fp32_host;
        CHECK_ACL(aclrtMallocHost(&probs_fp32_host, _info.n * sizeof(float)));

        CHECK_ACL(aclrtSynchronizeDevice());
        CHECK_ACL(aclrtMemcpy(probs_host, probs_host_size, probs, probs_host_size, ACL_MEMCPY_DEVICE_TO_HOST));

        auto fp32_ptr = static_cast<float *>(probs_fp32_host);
        if (_info.dt_p == INFINI_DTYPE_F16) {
            auto f16_ptr = static_cast<fp16_t *>(probs_host);
            for (uint64_t i = 0; i < _info.n; i++) {
                fp32_ptr[i] = _f16_to_f32(f16_ptr[i]);
            }
        } else {
            auto bf16_ptr = static_cast<bf16_t *>(probs_host);
            for (uint64_t i = 0; i < _info.n; i++) {
                fp32_ptr[i] = _bf16_to_f32(bf16_ptr[i]);
            }
        }

        CHECK_ACL(aclrtMemcpy(probs_fp32, probs_fp32_size, probs_fp32_host, probs_fp32_size, ACL_MEMCPY_HOST_TO_DEVICE));

        CHECK_ACL(aclrtFreeHost(probs_host));
        CHECK_ACL(aclrtFreeHost(probs_fp32_host));

        int64_t shape = _info.n;
        int64_t stride = 1;
        auto probs_fp32_desc = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_F32), {shape}, {stride});

        auto topk_val_fp32_desc = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_F32), topk_shape, topk_stride);
        auto topk_idx_desc = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_I64), topk_shape, topk_stride);

        CHECK_ACL(aclnnTopkGetWorkspaceSize(probs_fp32_desc->tensor,
                                            topk_shape[0],
                                            0,
                                            true,
                                            true,
                                            topk_val_fp32_desc->tensor,
                                            topk_idx_desc->tensor,
                                            &topk_workspace_size,
                                            &topk_executor));
        CHECK_ACL(aclSetAclOpExecutorRepeatable(topk_executor));
        void *topk_workspace;
        CHECK_ACL(aclrtMalloc(&topk_workspace, topk_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
        AclSetTensorAddr(topk_executor, 0, probs_fp32_desc->tensor, probs_fp32);
        AclSetTensorAddr(topk_executor, 1, topk_val_fp32_desc->tensor, topk_val_addr);
        AclSetTensorAddr(topk_executor, 2, topk_idx_desc->tensor, topk_idx_addr);
        CHECK_ACL(aclnnTopk(topk_workspace, topk_workspace_size, topk_executor, stream));
        CHECK_ACL(aclrtSynchronizeDevice());
        CHECK_ACL(aclrtFree(topk_workspace));

        delete topk_val_fp32_desc;
        delete topk_idx_desc;
        delete probs_fp32_desc;

        auto status = random_sample_kernel_launch(probs_fp32, result, topk_val_addr, topk_idx_addr, random_val, topp, effective_topk, temperature, _info.n, INFINI_DTYPE_F32, _info.dt_i, stream);
        CHECK_STATUS(status);
        CHECK_ACL(aclrtSynchronizeDevice());
        CHECK_ACL(aclrtFree(probs_fp32));
    } else {
        auto topk_val = new aclnnTensorDescriptor(toAclDataType(_info.dt_p), topk_shape, topk_stride);
        auto topk_idx = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_I64), topk_shape, topk_stride);

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
        CHECK_ACL(aclrtSynchronizeDevice());
        CHECK_ACL(aclrtFree(topk_workspace));

        auto status = random_sample_kernel_launch(probs_for_topk, result, topk_val_addr, topk_idx_addr, random_val, topp, effective_topk, temperature, _info.n, _info.dt_p, _info.dt_i, stream);
        CHECK_STATUS(status);

        delete topk_val;
        delete topk_idx;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::random_sample::ascend
