#include "../../operator.h"
#include "../../../utils.h"
#include "../../../utils/check.h"
#include "../../handle.h"
#include "../../tensor.h"
#include "infiniop/ops/bilinear.h"
#include "infiniop/ops/gemm.h"
#include "infiniop/ops/add.h"
#include "infiniop/ops/rearrange.h"

#include <algorithm>

struct InfiniopBilinearDescriptor {
    InfiniopDescriptor _super;
    infiniopGemmDescriptor_t imediate_desc;
    infiniopGemmDescriptor_t result_desc;
    infiniopRearrangeDescriptor_t weight_rearrange_desc;
    infiniopAddDescriptor_t bias_add_desc;
    size_t workspace_size;
    size_t weight_offset;
    size_t weight_size;
    size_t imediate_offset;
    size_t imediate_size;
    size_t op_workspace_offset;
    size_t op_workspace_size;
};

namespace {
constexpr size_t kWorkspaceAlignment = 256;

size_t aligned_size(size_t value) {
    return utils::align(value, kWorkspaceAlignment);
}
} // namespace

__C __export infiniStatus_t infiniopCreateBilinearDescriptor(
    infiniopHandle_t handle,
    infiniopBilinearDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc) {

    if (out_desc->ndim() != 2 || x1_desc->ndim() != 2 || x2_desc->ndim() != 2 || weight_desc->ndim() != 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t batch_size = x1_desc->shape()[0];
    size_t in1_features = x1_desc->shape()[1];
    size_t in2_features = x2_desc->shape()[1];
    size_t out_features = out_desc->shape()[1];

    if (x2_desc->shape()[0] != batch_size ||
        weight_desc->shape()[0] != out_features ||
        weight_desc->shape()[1] != in1_features ||
        weight_desc->shape()[2] != in2_features ||
        out_desc->shape()[0] != batch_size) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto dtype = out_desc->dtype();
    CHECK_OR_RETURN(x1_desc->dtype() == dtype && x2_desc->dtype() == dtype && weight_desc->dtype() == dtype,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);

    if (bias_desc) {
        CHECK_OR_RETURN(bias_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(bias_desc->ndim() == 1 && bias_desc->dim(0) == out_features,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    size_t dtype_size = infiniSizeOf(dtype);

    // Prepare packed weight layout to allow reshape into GEMM matrix
    size_t weight_shape[3] = {out_features, in1_features, in2_features};
    ptrdiff_t packed_strides[3] = {
        static_cast<ptrdiff_t>(in2_features),
        static_cast<ptrdiff_t>(out_features * in2_features),
        static_cast<ptrdiff_t>(1),
    };

    infiniopTensorDescriptor_t weight_packed_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&weight_packed_desc, 3, weight_shape, packed_strides, dtype));

    infiniopRearrangeDescriptor_t weight_rearrange_desc;
    CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &weight_rearrange_desc, weight_packed_desc, weight_desc));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(weight_packed_desc));

    infiniopTensorDescriptor_t weight_matrix_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&weight_matrix_desc, 3, weight_shape, packed_strides, dtype));
    TRANSFORM_TENSOR_DESC(weight_matrix_desc, dimPermute({1, 0, 2}));
    TRANSFORM_TENSOR_DESC(weight_matrix_desc, dimMerge(1, 2));

    size_t imediate_shape[2] = {batch_size, out_features * in2_features};
    infiniopTensorDescriptor_t imediate_flat_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&imediate_flat_desc, 2, imediate_shape, nullptr, dtype));

    infiniopGemmDescriptor_t imediate_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &imediate_desc, imediate_flat_desc, x1_desc, weight_matrix_desc));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(weight_matrix_desc));

    infiniopTensorDescriptor_t imediate_split_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&imediate_split_desc, 2, imediate_shape, nullptr, dtype));
    TRANSFORM_TENSOR_DESC(imediate_split_desc, dimSplit(1, {out_features, in2_features}));

    auto x2_shape = x2_desc->shape();
    auto x2_strides = x2_desc->strides();
    infiniopTensorDescriptor_t x2_col_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&x2_col_desc, 2, x2_shape.data(), x2_strides.data(), dtype));
    TRANSFORM_TENSOR_DESC(x2_col_desc, dimSplit(1, {in2_features, 1}));

    auto out_shape = out_desc->shape();
    auto out_strides = out_desc->strides();
    infiniopTensorDescriptor_t out_col_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&out_col_desc, 2, out_shape.data(), out_strides.data(), dtype));
    TRANSFORM_TENSOR_DESC(out_col_desc, dimSplit(1, {out_features, 1}));

    infiniopGemmDescriptor_t result_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &result_desc, out_col_desc, imediate_split_desc, x2_col_desc));

    size_t gemm1_workspace_size = 0;
    size_t gemm2_workspace_size = 0;
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(imediate_desc, &gemm1_workspace_size));
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(result_desc, &gemm2_workspace_size));

    CHECK_STATUS(infiniopDestroyTensorDescriptor(imediate_flat_desc));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(imediate_split_desc));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(x2_col_desc));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(out_col_desc));

    infiniopAddDescriptor_t bias_add_desc = nullptr;
    size_t add_workspace_size = 0;
    if (bias_desc) {
        size_t bias_shape[2] = {batch_size, out_features};
        ptrdiff_t bias_strides[2] = {0, bias_desc->stride(0)};
        infiniopTensorDescriptor_t bias_broadcast_desc;
        CHECK_STATUS(infiniopCreateTensorDescriptor(&bias_broadcast_desc, 2, bias_shape, bias_strides, dtype));
        CHECK_STATUS(infiniopCreateAddDescriptor(handle, &bias_add_desc, out_desc, out_desc, bias_broadcast_desc));
        CHECK_STATUS(infiniopGetAddWorkspaceSize(bias_add_desc, &add_workspace_size));
        CHECK_STATUS(infiniopDestroyTensorDescriptor(bias_broadcast_desc));
    }

    size_t weight_size = aligned_size(out_features * in1_features * in2_features * dtype_size);
    size_t imediate_size = aligned_size(batch_size * out_features * in2_features * dtype_size);

    size_t op_workspace_size = std::max(gemm1_workspace_size, gemm2_workspace_size);
    op_workspace_size = std::max(op_workspace_size, add_workspace_size);
    op_workspace_size = aligned_size(op_workspace_size);

    size_t weight_offset = 0;
    size_t imediate_offset = weight_offset + weight_size;
    size_t op_workspace_offset = imediate_offset + imediate_size;
    size_t workspace_size = op_workspace_offset + op_workspace_size;

    *(InfiniopBilinearDescriptor **)desc_ptr = new InfiniopBilinearDescriptor{
        {handle->device, handle->device_id},
        imediate_desc,
        result_desc,
        weight_rearrange_desc,
        bias_add_desc,
        workspace_size,
        weight_offset,
        weight_size,
        imediate_offset,
        imediate_size,
        op_workspace_offset,
        op_workspace_size};

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopGetBilinearWorkspaceSize(
    infiniopBilinearDescriptor_t desc,
    size_t *size) {
    *size = reinterpret_cast<InfiniopBilinearDescriptor *>(desc)->workspace_size;
    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopBilinear(
    infiniopBilinearDescriptor_t desc_,
    void *workspace,
    size_t workspace_size,
    void *out,
    void const *x1,
    void const *x2,
    void const *weight,
    void const *bias,
    void *stream) {

    auto desc = reinterpret_cast<InfiniopBilinearDescriptor *>(desc_);
    if (workspace_size < desc->workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (desc->bias_add_desc && bias == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (!desc->bias_add_desc && bias != nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    char *workspace_ptr = reinterpret_cast<char *>(workspace);
    void *weight_buffer = workspace_ptr + desc->weight_offset;
    void *imediate_buffer = workspace_ptr + desc->imediate_offset;
    void *op_workspace = workspace_ptr + desc->op_workspace_offset;

    CHECK_STATUS(infiniopRearrange(desc->weight_rearrange_desc, weight_buffer, weight, stream));

    CHECK_STATUS(infiniopGemm(desc->imediate_desc,
                              op_workspace, desc->op_workspace_size,
                              imediate_buffer,
                              x1,
                              weight_buffer,
                              1.0f,
                              0.0f,
                              stream));

    CHECK_STATUS(infiniopGemm(desc->result_desc,
                              op_workspace, desc->op_workspace_size,
                              out,
                              imediate_buffer,
                              x2,
                              1.0f,
                              0.0f,
                              stream));

    if (desc->bias_add_desc) {
        CHECK_STATUS(infiniopAdd(desc->bias_add_desc,
                                 op_workspace, desc->op_workspace_size,
                                 out,
                                 out,
                                 bias,
                                 stream));
    }

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopDestroyBilinearDescriptor(
    infiniopBilinearDescriptor_t desc_) {
    auto desc = reinterpret_cast<InfiniopBilinearDescriptor *>(desc_);

    CHECK_STATUS(infiniopDestroyGemmDescriptor(desc->imediate_desc));
    CHECK_STATUS(infiniopDestroyGemmDescriptor(desc->result_desc));
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(desc->weight_rearrange_desc));
    if (desc->bias_add_desc) {
        CHECK_STATUS(infiniopDestroyAddDescriptor(desc->bias_add_desc));
    }

    delete desc;
    return INFINI_STATUS_SUCCESS;
}
