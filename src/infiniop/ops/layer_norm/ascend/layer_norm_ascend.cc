#include "layer_norm_ascend.h"

#include "../../../devices/ascend/aclnn_executor.h"
#include <aclnnop/aclnn_layer_norm.h>
#include <aclnnop/aclnn_reciprocal.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace op::layer_norm::ascend {

namespace {

constexpr size_t ALIGNMENT = 32;

size_t align_up(size_t value) {
    return (value + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
}

std::vector<int64_t> to_i64(const std::vector<size_t> &values) {
    std::vector<int64_t> result;
    result.reserve(values.size());
    for (auto value : values) {
        result.push_back(static_cast<int64_t>(value));
    }
    return result;
}

std::vector<int64_t> to_i64(const std::vector<ptrdiff_t> &values) {
    std::vector<int64_t> result;
    result.reserve(values.size());
    for (auto value : values) {
        result.push_back(static_cast<int64_t>(value));
    }
    return result;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t> &shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    return strides;
}

aclnnTensorDescriptor_t make_tensor(
    aclDataType dtype,
    const std::vector<int64_t> &shape,
    const std::vector<int64_t> &strides) {
    return new aclnnTensorDescriptor(dtype, shape, strides, nullptr);
}

} // namespace

struct Descriptor::Opaque {
    device::ascend::AclnnExecutor standard;
    device::ascend::AclnnExecutor affine;
    device::ascend::AclnnExecutor reciprocal;
    aclIntArray *normalized_shape = nullptr;
    size_t acl_workspace_size = 0;
    size_t mean_offset = 0;
    size_t rstd_offset = 0;
    bool bias_exist = false;

    ~Opaque() {
        if (normalized_shape != nullptr) {
            aclDestroyIntArray(normalized_shape);
        }
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    float eps) {

    if (output_desc == nullptr || input_standardization_desc == nullptr
        || input_std_deviation_desc == nullptr || input_desc == nullptr
        || weight_desc == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    if (output_desc->dtype() != dtype
        || input_standardization_desc->dtype() != dtype
        || input_std_deviation_desc->dtype() != dtype
        || weight_desc->dtype() != dtype
        || (bias_desc != nullptr && bias_desc->dtype() != dtype)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = LayerNormInfo::createLayerNormInfo(
        output_desc, input_standardization_desc, input_std_deviation_desc,
        input_desc, weight_desc, bias_desc, eps);
    CHECK_RESULT(result);
    auto info = result.take();

    auto opaque = std::make_unique<Opaque>();
    opaque->bias_exist = bias_desc != nullptr;

    auto acl_dtype = toAclDataType(dtype);
    auto input_shape = to_i64(info.input_shape);
    auto keepdim_shape = input_shape;
    keepdim_shape.back() = 1;
    auto keepdim_strides = contiguous_strides(keepdim_shape);
    auto std_deviation_strides = to_i64(info.input_std_deviation_strides);
    std_deviation_strides.push_back(1);
    auto normalized_size = static_cast<int64_t>(info.normalized_size);
    opaque->normalized_shape = aclCreateIntArray(&normalized_size, 1);
    if (opaque->normalized_shape == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    auto &standard = opaque->standard;
    standard.tensors = {
        make_tensor(acl_dtype, input_shape, to_i64(info.input_strides)),
        make_tensor(
            acl_dtype, input_shape,
            to_i64(info.input_standardization_strides)),
        make_tensor(acl_dtype, keepdim_shape, keepdim_strides),
        make_tensor(acl_dtype, keepdim_shape, keepdim_strides),
    };
    CHECK_ACL(aclnnLayerNormGetWorkspaceSize(
        standard.tensors[0]->tensor,
        opaque->normalized_shape,
        nullptr,
        nullptr,
        static_cast<double>(eps),
        standard.tensors[1]->tensor,
        standard.tensors[2]->tensor,
        standard.tensors[3]->tensor,
        &standard.workspace_size,
        &standard.executor));
    aclSetAclOpExecutorRepeatable(standard.executor);

    auto &affine = opaque->affine;
    affine.tensors = {
        make_tensor(acl_dtype, input_shape, to_i64(info.input_strides)),
        make_tensor(
            acl_dtype, {normalized_size},
            to_i64(info.weight_strides)),
    };
    if (opaque->bias_exist) {
        affine.tensors.push_back(make_tensor(
            acl_dtype, {normalized_size}, to_i64(info.bias_strides)));
    }
    affine.tensors.push_back(
        make_tensor(acl_dtype, input_shape, to_i64(info.output_strides)));
    affine.tensors.push_back(
        make_tensor(acl_dtype, keepdim_shape, keepdim_strides));
    affine.tensors.push_back(
        make_tensor(acl_dtype, keepdim_shape, keepdim_strides));

    size_t affine_output_index = opaque->bias_exist ? 3 : 2;
    CHECK_ACL(aclnnLayerNormGetWorkspaceSize(
        affine.tensors[0]->tensor,
        opaque->normalized_shape,
        affine.tensors[1]->tensor,
        opaque->bias_exist ? affine.tensors[2]->tensor : nullptr,
        static_cast<double>(eps),
        affine.tensors[affine_output_index]->tensor,
        affine.tensors[affine_output_index + 1]->tensor,
        affine.tensors[affine_output_index + 2]->tensor,
        &affine.workspace_size,
        &affine.executor));
    aclSetAclOpExecutorRepeatable(affine.executor);

    auto &reciprocal = opaque->reciprocal;
    reciprocal.tensors = {
        make_tensor(acl_dtype, keepdim_shape, keepdim_strides),
        make_tensor(
            acl_dtype, keepdim_shape, std_deviation_strides),
    };
    CHECK_ACL(aclnnReciprocalGetWorkspaceSize(
        reciprocal.tensors[0]->tensor,
        reciprocal.tensors[1]->tensor,
        &reciprocal.workspace_size,
        &reciprocal.executor));
    aclSetAclOpExecutorRepeatable(reciprocal.executor);

    opaque->acl_workspace_size = std::max(
        standard.workspace_size,
        std::max(affine.workspace_size, reciprocal.workspace_size));
    size_t statistic_size = info.othersize * infiniSizeOf(dtype);
    opaque->mean_offset = align_up(opaque->acl_workspace_size);
    opaque->rstd_offset = align_up(opaque->mean_offset + statistic_size);
    size_t workspace_size = opaque->rstd_offset + statistic_size;

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    *desc_ptr = new Descriptor(
        dtype, std::move(info), workspace_size, opaque.release(),
        handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *input_standardization,
    void *input_std_deviation,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream) const {

    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    void *mean = static_cast<uint8_t *>(workspace) + _opaque->mean_offset;
    void *rstd = static_cast<uint8_t *>(workspace) + _opaque->rstd_offset;

    _opaque->standard.bind({
        const_cast<void *>(input),
        input_standardization,
        mean,
        rstd,
    });
    if (_opaque->bias_exist) {
        _opaque->affine.bind({
            const_cast<void *>(input),
            const_cast<void *>(weight),
            const_cast<void *>(bias),
            output,
            mean,
            rstd,
        });
    } else {
        _opaque->affine.bind({
            const_cast<void *>(input),
            const_cast<void *>(weight),
            output,
            mean,
            rstd,
        });
    }
    _opaque->reciprocal.bind({rstd, input_std_deviation});

    auto acl_stream = static_cast<aclrtStream>(stream);
    CHECK_ACL(aclnnLayerNorm(
        workspace, _opaque->standard.workspace_size,
        _opaque->standard.executor, acl_stream));
    CHECK_ACL(aclnnLayerNorm(
        workspace, _opaque->affine.workspace_size,
        _opaque->affine.executor, acl_stream));
    CHECK_ACL(aclnnReciprocal(
        workspace, _opaque->reciprocal.workspace_size,
        _opaque->reciprocal.executor, acl_stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::layer_norm::ascend
