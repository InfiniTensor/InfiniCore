#include "swiglu_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_swi_glu.h>

namespace op::swiglu::ascend {

// Opaque structure must be defined AFTER the class declaration (which is in swiglu.h via DESCRIPTOR macro)
struct Descriptor::Opaque {
    aclnnTensorDescriptor_t x;        // Combined input tensor (gate and up concatenated)
    int64_t dim;                       // Dimension along which to split
    aclnnTensorDescriptor_t out;      // Output tensor
    size_t workspaceSize;
    aclOpExecutor *executor;

    Opaque(aclnnTensorDescriptor_t x_, int64_t dim_, aclnnTensorDescriptor_t out_,
           size_t ws, aclOpExecutor *exec)
        : x(x_), dim(dim_), out(out_), workspaceSize(ws), executor(exec) {}

    ~Opaque() {
        delete x;
        delete out;
        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

// Helper function to create a combined tensor descriptor from gate and up descriptors
static aclnnTensorDescriptor_t createCombinedTensor(
    infiniopTensorDescriptor_t gate_desc,
    infiniopTensorDescriptor_t up_desc,
    int64_t dim) {
    
    auto gate_shape = gate_desc->shape();
    auto ndim = gate_desc->ndim();
    
    // Create combined shape
    std::vector<int64_t> combined_shape(gate_shape.begin(), gate_shape.end());
    combined_shape[dim] *= 2;
    
    // Calculate strides for combined tensor
    std::vector<int64_t> combined_strides(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        combined_strides[i] = combined_shape[i + 1] * combined_strides[i + 1];
    }
    
    // Map dtype
    aclDataType acl_dtype;
    switch (gate_desc->dtype()) {
        case INFINI_DTYPE_F16:
            acl_dtype = ACL_FLOAT16;
            break;
        case INFINI_DTYPE_F32:
            acl_dtype = ACL_FLOAT;
            break;
        case INFINI_DTYPE_BF16:
            acl_dtype = ACL_BF16;
            break;
        default:
            return nullptr;
    }
    
    return new aclnnTensorDescriptor(acl_dtype, combined_shape, combined_strides);
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    std::vector<infiniopTensorDescriptor_t> input_descs) {

    // SwiGLU in ACL expects a single input tensor that will be split along dim
    if (input_descs.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto gate_desc = input_descs[0];
    auto up_desc = input_descs[1];

    // Create SwiGLUInfo first
    auto result = SwiGLUInfo::create(output_desc, gate_desc, up_desc);
    CHECK_RESULT(result);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    auto ndim = gate_desc->ndim();
    int64_t dim = static_cast<int64_t>(ndim) - 1;  // Split along the last dimension by default
    
    // Create combined tensor descriptor
    aclnnTensorDescriptor_t x = createCombinedTensor(gate_desc, up_desc, dim);
    if (!x) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    aclnnTensorDescriptor_t out = new aclnnTensorDescriptor(output_desc);

    size_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    CHECK_ACL(aclnnSwiGluGetWorkspaceSize(
        x->tensor,
        dim,
        out->tensor,
        &workspace_size,
        &executor));

    aclSetAclOpExecutorRepeatable(executor);

    *desc_ptr = new Descriptor(
        new Opaque{x, dim, out, workspace_size, executor},
        result.take(),
        workspace_size,
        handle_ascend->device,
        handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output, std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    // In ACL's SwiGLU, the input is a single tensor where gate and up are concatenated
    // inputs[0] should point to the start of the combined gate+up tensor
    // inputs[1] should point to the up tensor (second half of combined tensor)
    // If the data is not already concatenated, this needs to be handled by the caller
    
    AclSetTensorAddr(_opaque->executor, 0, _opaque->x->tensor, const_cast<void *>(inputs[0]));
    AclSetTensorAddr(_opaque->executor, 1, _opaque->out->tensor, output);

    CHECK_ACL(aclnnSwiGlu(
        workspace,
        workspace_size,
        _opaque->executor,
        stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::swiglu::ascend
