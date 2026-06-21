#include "causal_softmax_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_masked_fill_tensor.h>
#include <aclnnop/aclnn_softmax.h>
#include <algorithm>

namespace op::causal_softmax::ascend {

namespace {

bool isCompact(const CausalSoftmaxInfo &info, ptrdiff_t stride_b, ptrdiff_t stride_i, ptrdiff_t stride_j) {
    return stride_j == 1
        && stride_i == static_cast<ptrdiff_t>(info.total_seq_len)
        && (info.batch_size == 1 || stride_b == static_cast<ptrdiff_t>(info.seq_len * info.total_seq_len));
}

} // namespace

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t x;
    aclnnTensorDescriptor_t mask;
    aclnnTensorDescriptor_t y;
    aclnnTensorDescriptor_t value;
    aclnnTensorDescriptor_t temp_x;
    aclnnTensorDescriptor_t temp_y;
    void *mask_addr;
    void *value_addr;
    void *temp_x_addr;
    void *temp_y_addr;
    size_t workspacesize;
    aclOpExecutor *executor;
    aclOpExecutor *temp_executor;
    aclOpExecutor *copy_in_executor;
    aclOpExecutor *copy_out_executor;
    bool use_temp;

    ~Opaque() {
        delete x;
        delete mask;
        delete y;
        delete value;
        delete temp_x;
        delete temp_y;

        aclrtFree(mask_addr);
        aclrtFree(value_addr);
        aclrtFree(temp_x_addr);
        aclrtFree(temp_y_addr);

        // Delete useless executor
        aclDestroyAclOpExecutor(executor);
        aclDestroyAclOpExecutor(temp_executor);
        aclDestroyAclOpExecutor(copy_in_executor);
        aclDestroyAclOpExecutor(copy_out_executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    auto result = CausalSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(result);
    CausalSoftmaxInfo info = result.take();

    aclOpExecutor *executor = nullptr;
    aclOpExecutor *temp_executor = nullptr;
    aclOpExecutor *mask_executor = nullptr;
    aclOpExecutor *copy_in_executor = nullptr;
    aclOpExecutor *copy_out_executor = nullptr;
    aclnnTensorDescriptor_t y = nullptr;
    aclnnTensorDescriptor_t mask = nullptr;
    aclnnTensorDescriptor_t x = nullptr;
    aclnnTensorDescriptor_t value = nullptr;
    aclnnTensorDescriptor_t temp_x = nullptr;
    aclnnTensorDescriptor_t temp_y = nullptr;
    void *mask_addr = nullptr;
    void *value_addr = nullptr;
    void *temp_x_addr = nullptr;
    void *temp_y_addr = nullptr;
    size_t workspacesize_softmax = 0;
    size_t workspacesize_temp_softmax = 0;
    size_t workspacesize_mask = 0;
    size_t workspacesize_copy_in = 0;
    size_t workspacesize_copy_out = 0;

    // Create Aclnn Tensor Descriptors for input, mask and output
    std::vector<int64_t> shape = {static_cast<int64_t>(info.batch_size), static_cast<int64_t>(info.seq_len), static_cast<int64_t>(info.total_seq_len)};
    std::vector<int64_t> x_strides = {static_cast<int64_t>(info.x_stride_b), static_cast<int64_t>(info.x_stride_i), static_cast<int64_t>(info.x_stride_j)};
    std::vector<int64_t> y_strides = {static_cast<int64_t>(info.y_stride_b), static_cast<int64_t>(info.y_stride_i), static_cast<int64_t>(info.y_stride_j)};
    std::vector<int64_t> compact_strides = {static_cast<int64_t>(info.seq_len * info.total_seq_len), static_cast<int64_t>(info.total_seq_len), 1};
    y = new aclnnTensorDescriptor(toAclDataType(info.dtype), shape, y_strides);
    x = new aclnnTensorDescriptor(toAclDataType(info.dtype), shape, x_strides);
    temp_x = new aclnnTensorDescriptor(toAclDataType(info.dtype), shape, compact_strides);
    temp_y = new aclnnTensorDescriptor(toAclDataType(info.dtype), shape, compact_strides);
    mask = new aclnnTensorDescriptor(aclDataType::ACL_BOOL, {static_cast<int64_t>(info.seq_len), static_cast<int64_t>(info.total_seq_len)}, {static_cast<int64_t>(info.total_seq_len), 1});

    // Initialize the value tensor with -inf
    if (info.dtype == INFINI_DTYPE_F16) {
        uint16_t mask_value = 0xfc00;
        auto size = aclDataTypeSize(aclDataType::ACL_FLOAT16);
        CHECK_ACL(aclrtMalloc(&value_addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(value_addr, size, &mask_value, size, ACL_MEMCPY_HOST_TO_DEVICE));
        value = new aclnnTensorDescriptor(aclDataType::ACL_FLOAT16, {}, {});
    } else {
        uint32_t mask_value = 0xff800000;
        auto size = aclDataTypeSize(aclDataType::ACL_FLOAT);
        CHECK_ACL(aclrtMalloc(&value_addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(value_addr, size, &mask_value, size, ACL_MEMCPY_HOST_TO_DEVICE));
        value = new aclnnTensorDescriptor(aclDataType::ACL_FLOAT, {}, {});
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
    aclTensor *ttemp_x = temp_x->tensor;
    aclTensor *ttemp_y = temp_y->tensor;
    aclTensor *tmask = mask->tensor;
    aclTensor *tvalue = value->tensor;

    bool use_temp = !isCompact(info, info.x_stride_b, info.x_stride_i, info.x_stride_j)
                 || !isCompact(info, info.y_stride_b, info.y_stride_i, info.y_stride_j);

    if (use_temp) {
        CHECK_ACL(aclnnInplaceCopyGetWorkspaceSize(ttemp_x, tx, &workspacesize_copy_in, &copy_in_executor));
        aclSetAclOpExecutorRepeatable(copy_in_executor);
        CHECK_ACL(aclnnInplaceCopyGetWorkspaceSize(ty, ttemp_y, &workspacesize_copy_out, &copy_out_executor));
        aclSetAclOpExecutorRepeatable(copy_out_executor);
        CHECK_ACL(aclnnInplaceMaskedFillTensorGetWorkspaceSize(ttemp_x, tmask, tvalue, &workspacesize_mask, &mask_executor));
        int64_t dim = 2;
        CHECK_ACL(aclnnSoftmaxGetWorkspaceSize(ttemp_x, dim, ttemp_y, &workspacesize_temp_softmax, &temp_executor));
        aclSetAclOpExecutorRepeatable(temp_executor);
    } else {
        CHECK_ACL(aclnnInplaceMaskedFillTensorGetWorkspaceSize(tx, tmask, tvalue, &workspacesize_mask, &mask_executor));
        int64_t dim = 2;
        CHECK_ACL(aclnnSoftmaxGetWorkspaceSize(tx, dim, ty, &workspacesize_softmax, &executor));
        // set executor reusable
        aclSetAclOpExecutorRepeatable(executor);
    }

    size_t op_workspace_size = std::max(std::max(workspacesize_softmax, workspacesize_temp_softmax),
                                        std::max(workspacesize_mask, std::max(workspacesize_copy_in, workspacesize_copy_out)));
    size_t all_workspacesize = op_workspace_size;
    if (use_temp) {
        size_t temp_bytes = temp_x->numel() * infiniSizeOf(info.dtype);
        CHECK_ACL(aclrtMalloc(&temp_x_addr, temp_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&temp_y_addr, temp_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    *desc_ptr = new Descriptor(new Opaque{x, mask, y, value, temp_x, temp_y, mask_addr, value_addr,
                                          temp_x_addr, temp_y_addr, op_workspace_size, executor, temp_executor, copy_in_executor, copy_out_executor, use_temp},
                               std::move(info), all_workspacesize, handle_ascend->device, handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *y, const void *x, void *stream) const {
    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto tx = _opaque->x->tensor;
    auto ty = _opaque->y->tensor;
    auto tmask = _opaque->mask->tensor;
    auto tvalue = _opaque->value->tensor;

    if (_opaque->use_temp) {
        auto ttemp_x = _opaque->temp_x->tensor;
        auto ttemp_y = _opaque->temp_y->tensor;
        void *temp_x = _opaque->temp_x_addr;
        void *temp_y = _opaque->temp_y_addr;

        AclSetTensorAddr(_opaque->copy_in_executor, 0, ttemp_x, temp_x);
        AclSetTensorAddr(_opaque->copy_in_executor, 1, tx, (void *)x);
        CHECK_ACL(aclnnInplaceCopy(workspace, _opaque->workspacesize, _opaque->copy_in_executor, stream));

        aclOpExecutor *mask_executor = nullptr;
        size_t workspacesize_mask = 0;
        AclSetTensorAddr(mask_executor, 0, ttemp_x, temp_x);
        AclSetTensorAddr(mask_executor, 1, tmask, _opaque->mask_addr);
        AclSetTensorAddr(mask_executor, 2, tvalue, _opaque->value_addr);
        CHECK_ACL(aclnnInplaceMaskedFillTensorGetWorkspaceSize(ttemp_x, tmask, tvalue, &workspacesize_mask, &mask_executor));
        CHECK_ACL(aclnnInplaceMaskedFillTensor(workspace, _opaque->workspacesize, mask_executor, stream));

        AclSetTensorAddr(_opaque->temp_executor, 0, ttemp_x, temp_x);
        AclSetTensorAddr(_opaque->temp_executor, 1, ttemp_y, temp_y);
        CHECK_ACL(aclnnSoftmax(workspace, _opaque->workspacesize, _opaque->temp_executor, stream));

        AclSetTensorAddr(_opaque->copy_out_executor, 0, ty, y);
        AclSetTensorAddr(_opaque->copy_out_executor, 1, ttemp_y, temp_y);
        CHECK_ACL(aclnnInplaceCopy(workspace, _opaque->workspacesize, _opaque->copy_out_executor, stream));
        return INFINI_STATUS_SUCCESS;
    }

    aclOpExecutor *mask_executor = nullptr;
    size_t workspacesize_mask = 0;

    AclSetTensorAddr(mask_executor, 0, tx, (void *)x);
    AclSetTensorAddr(mask_executor, 1, tmask, _opaque->mask_addr);
    AclSetTensorAddr(mask_executor, 2, tvalue, _opaque->value_addr);
    CHECK_ACL(aclnnInplaceMaskedFillTensorGetWorkspaceSize(tx, tmask, tvalue, &workspacesize_mask, &mask_executor));
    CHECK_ACL(aclnnInplaceMaskedFillTensor(workspace, _opaque->workspacesize, mask_executor, stream));

    AclSetTensorAddr(_opaque->executor, 0, tx, (void *)x);
    AclSetTensorAddr(_opaque->executor, 1, ty, y);
    CHECK_ACL(aclnnSoftmax(workspace, _opaque->workspacesize, _opaque->executor, stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::ascend
