#include "silu_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_silu.h>

namespace op::silu::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t c, a;
    aclOpExecutor *executor;

    ~Opaque() {
        delete c;
        delete a;
        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    std::array<infiniopTensorDescriptor_t, 1> a_desc) {
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto t_c = new aclnnTensorDescriptor(c_desc);
    auto t_a = new aclnnTensorDescriptor(a_desc[0]);

    auto tc = t_c->tensor,
         ta = t_a->tensor;

    aclOpExecutor *executor = nullptr;
    uint64_t workspace_size = 0;
    // Get workspace size and executor
    CHECK_ACL(aclnnSiluGetWorkspaceSize(ta, tc, &workspace_size, &executor));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(executor));

    *desc_ptr = new Descriptor(
        dtype, workspace_size,
        new Opaque{t_c, t_a, executor},
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    std::array<const void *, 1> a,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    aclSetTensorAddr(_opaque->executor, 0, _opaque->a->tensor, (void *)a[0]);
    aclSetTensorAddr(_opaque->executor, 1, _opaque->c->tensor, c);

    CHECK_ACL(aclnnSilu(workspace, workspace_size, _opaque->executor, stream));
    // aclrtSynchronizeStream(stream);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::silu::ascend
