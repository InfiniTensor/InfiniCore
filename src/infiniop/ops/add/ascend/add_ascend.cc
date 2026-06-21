#include "add_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_add.h>
#include <cstdint>
#include <cstdlib>

namespace op::add::ascend {

// Opaque structure must be defined AFTER the class declaration (which is in add.h via DESCRIPTOR macro)
struct Descriptor::Opaque {
    aclnnTensorDescriptor_t a;
    aclnnTensorDescriptor_t b;
    aclnnTensorDescriptor_t c;
    aclnnScalarDescriptor_t alpha;
    void *alpha_value;
    size_t workspaceSize;
    aclOpExecutor *executor;

    Opaque(aclnnTensorDescriptor_t a_, aclnnTensorDescriptor_t b_, aclnnTensorDescriptor_t c_,
           aclnnScalarDescriptor_t alpha_, void *alpha_value_, size_t ws, aclOpExecutor *exec)
        : a(a_), b(b_), c(c_), alpha(alpha_), alpha_value(alpha_value_), workspaceSize(ws), executor(exec) {}

    ~Opaque() {
        delete a;
        delete b;
        delete c;
        delete alpha;
        std::free(alpha_value);
        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    std::vector<infiniopTensorDescriptor_t> input_descs) {

    if (input_descs.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto a_desc = input_descs[0];
    auto b_desc = input_descs[1];

    // Create AddInfo first
    auto result = AddInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    aclnnTensorDescriptor_t a = new aclnnTensorDescriptor(a_desc);
    aclnnTensorDescriptor_t b = new aclnnTensorDescriptor(b_desc);
    aclnnTensorDescriptor_t c = new aclnnTensorDescriptor(c_desc);

    void *alpha_value = nullptr;
    size_t alpha_value_size = 0;
    infiniDtype_t alpha_dtype = INFINI_DTYPE_F32;

#define SET_ALPHA(TYPE, DTYPE, VALUE)                      \
    do {                                                   \
        alpha_value_size = sizeof(TYPE);                   \
        alpha_value = std::malloc(alpha_value_size);       \
        if (alpha_value == nullptr) {                      \
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;   \
        }                                                  \
        *static_cast<TYPE *>(alpha_value) = (TYPE)(VALUE); \
        alpha_dtype = DTYPE;                               \
    } while (0)

    switch (c_desc->dtype()) {
    case INFINI_DTYPE_I32:
        SET_ALPHA(int32_t, INFINI_DTYPE_I32, 1);
        break;
    case INFINI_DTYPE_I64:
        SET_ALPHA(int64_t, INFINI_DTYPE_I64, 1);
        break;
    default:
        SET_ALPHA(float, INFINI_DTYPE_F32, 1.0f);
        break;
    }

#undef SET_ALPHA

    aclnnScalarDescriptor_t alpha = new aclnnScalarDescriptor(
        alpha_dtype, alpha_value, alpha_value_size);

    size_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    CHECK_ACL(aclnnAddGetWorkspaceSize(
        a->tensor,
        b->tensor,
        alpha->scalar,
        c->tensor,
        &workspace_size,
        &executor));

    aclSetAclOpExecutorRepeatable(executor);

    *desc_ptr = new Descriptor(
        new Opaque{a, b, c, alpha, alpha_value, workspace_size, executor},
        result.take(),
        workspace_size,
        handle_ascend->device,
        handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *c, std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    // Set input/output tensor addresses in the executor
    // Parameters: executor, tensor_index, tensor_descriptor, data_pointer
    AclSetTensorAddr(_opaque->executor, 0, _opaque->a->tensor, const_cast<void *>(inputs[0]));
    AclSetTensorAddr(_opaque->executor, 1, _opaque->b->tensor, const_cast<void *>(inputs[1]));
    AclSetTensorAddr(_opaque->executor, 2, _opaque->c->tensor, c);

    CHECK_ACL(aclnnAdd(
        workspace,
        workspace_size,
        _opaque->executor,
        stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::add::ascend
