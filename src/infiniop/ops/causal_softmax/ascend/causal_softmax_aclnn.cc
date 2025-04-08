#include "causal_softmax_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_masked_fill_tensor.h>
#include <aclnnop/aclnn_softmax.h>

namespace op::causal_softmax::ascend {

struct Descriptor::Opaque {
    mutable aclOpExecutor *executor;
    mutable aclOpExecutor *mask_executor;
    aclnnTensorDescriptor_t x;
    aclnnTensorDescriptor_t mask;
    aclnnTensorDescriptor_t y;
    void *mask_addr;
    size_t workspacesize_softmax;
    size_t workspacesize_mask;

    ~Opaque() {
        delete x;
        delete mask;
        delete y;

        aclDestroyAclOpExecutor(executor);
        aclDestroyAclOpExecutor(mask_executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc) {
    CausalSoftmaxInfo info;
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    CHECK_STATUS(createCausalSoftmaxInfo(&info, y_desc));

    aclOpExecutor *executor = nullptr;
    aclOpExecutor *mask_executor = nullptr;
    aclnnTensorDescriptor_t y = nullptr;
    aclnnTensorDescriptor_t mask = nullptr;
    aclnnTensorDescriptor_t x = nullptr;
    aclnnTensorDescriptor_t value = nullptr;
    void *mask_addr = nullptr;
    void *valueAddr = nullptr;
    size_t workspacesize_softmax = 0;
    size_t workspacesize_mask = 0;

    // Create Aclnn Tensor Descriptors for input , mask and output
    std::vector<int64_t> shape = {static_cast<int64_t>(info.batch_size), static_cast<int64_t>(info.seq_len), static_cast<int64_t>(info.total_seq_len)};
    std::vector<int64_t> strides = {static_cast<int64_t>(info.stride_b), static_cast<int64_t>(info.stride_i), static_cast<int64_t>(info.stride_j)};
    y = new aclnnTensorDescriptor(toAclDataType(info.dtype), shape, strides);
    x = new aclnnTensorDescriptor(toAclDataType(info.dtype), shape, strides);
    mask = new aclnnTensorDescriptor(aclDataType::ACL_BOOL, {static_cast<int64_t>(info.seq_len), static_cast<int64_t>(info.total_seq_len)}, {static_cast<int64_t>(info.total_seq_len), 1});

    // Initialize the value tensor with -∞
    if (info.dtype == INFINI_DTYPE_F16) {
        uint16_t mask_value = 0xfc00;
        auto size = aclDataTypeSize(aclDataType::ACL_FLOAT16);
        CHECK_ACL(aclrtMalloc(&valueAddr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(valueAddr, size, &mask_value, size, ACL_MEMCPY_HOST_TO_DEVICE));
        value = new aclnnTensorDescriptor(aclDataType::ACL_FLOAT16, {}, {}, valueAddr);
    } else {
        uint32_t mask_value = 0xff800000;
        auto size = aclDataTypeSize(aclDataType::ACL_FLOAT);
        CHECK_ACL(aclrtMalloc(&valueAddr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(valueAddr, size, &mask_value, size, ACL_MEMCPY_HOST_TO_DEVICE));
        value = new aclnnTensorDescriptor(aclDataType::ACL_FLOAT, {}, {}, valueAddr);
    }

    // Fill Mask Tensor
    std::vector<char> mask_matrix(mask->numel(), 0);
    for (size_t i = 0; i < info.seq_len; ++i) {
        for (size_t j = info.total_seq_len - info.seq_len + i + 1; j < info.total_seq_len; ++j) {
            size_t index = i * info.total_seq_len + j;
            mask_matrix[index] = 1;
        }
    }

    auto size = mask->numel() * aclDataTypeSize(aclDataType::ACL_BOOL);
    CHECK_ACL(aclrtMalloc(&mask_addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(mask_addr, size, mask_matrix.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));

    // Get the workspace size for the op
    aclTensor *tx = x->tensor;
    aclTensor *ty = y->tensor;
    aclTensor *tmask = mask->tensor;
    aclTensor *tvalue = value->tensor;

    CHECK_ACL(aclnnInplaceMaskedFillTensorGetWorkspaceSize(tx, tmask, tvalue, &workspacesize_mask, &mask_executor));
    aclSetAclOpExecutorRepeatable(mask_executor);
    int64_t dim = 2;

    CHECK_ACL(aclnnSoftmaxGetWorkspaceSize(tx, dim, ty, &workspacesize_softmax, &executor));
    aclSetAclOpExecutorRepeatable(executor);

    // Create the descriptor
    size_t all_workspacesize = workspacesize_softmax + workspacesize_mask;
    *desc_ptr = new Descriptor(new Opaque{executor, mask_executor, x, mask, y, mask_addr, workspacesize_softmax, workspacesize_mask},
                               info, all_workspacesize, handle_ascend->device, handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *data, void *stream) {
    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto tx = _opaque->x->tensor;
    auto ty = _opaque->y->tensor;
    auto tmask = _opaque->mask->tensor;
    auto executor = _opaque->executor;
    auto mask_executor = _opaque->mask_executor;
    auto mask_addr = _opaque->mask_addr;

    AclSetTensorAddr(mask_executor, 0, tx, data);
    AclSetTensorAddr(mask_executor, 1, tmask, mask_addr);
    CHECK_ACL(aclnnInplaceMaskedFillTensor(workspace, _opaque->workspacesize_mask, mask_executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    AclSetTensorAddr(executor, 0, tx, data);
    AclSetTensorAddr(executor, 2, ty, data);
    CHECK_ACL(aclnnSoftmax(workspace, _opaque->workspacesize_softmax, executor, stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::ascend
