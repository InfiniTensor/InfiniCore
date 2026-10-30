#include "rearrange_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_copy.h>

namespace op::rearrange::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t dst;
    aclnnTensorDescriptor_t src;
    void *workspace; // aclnnInplaceCopy workspace
    uint64_t workspace_size;
    // Layout metadata for calculate(): descriptors rebuilt per call with the
    // CURRENT data pointers (the create-time aclTensors carry no address).
    bool trivial = false; // src and dst are both row-major contiguous
    size_t nbytes = 0;
    aclDataType acl_dtype = ACL_FLOAT;
    std::vector<int64_t> shape;
    std::vector<int64_t> dst_strides;
    std::vector<int64_t> src_strides;
    ~Opaque() {
        delete dst;
        delete src;

        aclrtFree(workspace);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
};

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto dtype = y_desc->dtype();
    auto ndim = y_desc->ndim();
    auto shape = y_desc->shape();
    CHECK_API_OR(x_desc->dtype(), dtype, return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(x_desc->ndim(), ndim, return INFINI_STATUS_BAD_TENSOR_SHAPE);

    for (size_t i = 0; i < ndim; ++i) {
        CHECK_API_OR(x_desc->shape()[i], shape[i], return INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    auto dst_strides = y_desc->strides();
    auto src_strides = x_desc->strides();
    auto element_size = infiniSizeOf(dtype);

    auto result = utils::RearrangeMeta::create(shape.data(), dst_strides.data(), src_strides.data(), ndim, element_size);
    CHECK_RESULT(result);

    aclnnTensorDescriptor_t dst = new aclnnTensorDescriptor(y_desc);
    aclnnTensorDescriptor_t src = new aclnnTensorDescriptor(x_desc);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    void *workspace = nullptr;
    aclnnInplaceCopyGetWorkspaceSize(dst->tensor, src->tensor,
                                     &workspace_size, &executor);
    if (workspace_size != 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    auto opaque = new Opaque{
        dst,
        src,
        workspace,
        workspace_size};

    // Detect the trivial contiguous case: both strides row-major.
    {
        int64_t acc = 1;
        opaque->trivial = true;
        for (size_t i = ndim; i-- > 0;) {
            if (static_cast<int64_t>(dst_strides[i]) != acc ||
                static_cast<int64_t>(src_strides[i]) != acc) {
                opaque->trivial = false;
            }
            acc *= static_cast<int64_t>(shape[i]);
        }
        opaque->nbytes = static_cast<size_t>(acc) * element_size;
        opaque->acl_dtype = toAclDataType(dtype);
        opaque->shape.assign(shape.begin(), shape.end());
        opaque->dst_strides.assign(dst_strides.begin(), dst_strides.end());
        opaque->src_strides.assign(src_strides.begin(), src_strides.end());
    }

    *desc_ptr = new Descriptor(
        result.take(),
        opaque,
        handle->device,
        handle->device_id);

    // Delete useless executor
    aclDestroyAclOpExecutor(executor);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {
    // Fast path: identical contiguous layouts -> plain D2D memcpy.
    if (_opaque->trivial) {
        CHECK_ACL(aclrtMemcpyAsync(y, _opaque->nbytes, x, _opaque->nbytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   static_cast<aclrtStream>(stream)));
        return INFINI_STATUS_SUCCESS;
    }

    // General path: fresh descriptors carrying the CURRENT addresses
    // (same pattern as the working rotary-embedding copy kernel).
    aclnnTensorDescriptor tdst(_opaque->acl_dtype, _opaque->shape,
                               _opaque->dst_strides, y);
    aclnnTensorDescriptor tsrc(_opaque->acl_dtype, _opaque->shape,
                               _opaque->src_strides, const_cast<void *>(x));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnInplaceCopyGetWorkspaceSize(tdst.tensor, tsrc.tensor,
                                               &workspace_size, &executor));
    void *workspace = _opaque->workspace;
    if (workspace_size > _opaque->workspace_size) {
        CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnInplaceCopy(workspace, workspace_size, executor, stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rearrange::ascend
