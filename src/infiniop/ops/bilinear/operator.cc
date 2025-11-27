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
    infiniopGemmDescriptor_t gemm_1_desc;
    infiniopGemmDescriptor_t gemm_2_desc;
    infiniopRearrangeDescriptor_t weight_rearrange_desc;
    infiniopRearrangeDescriptor_t x1_rearrange_desc;
    infiniopRearrangeDescriptor_t x2_rearrange_desc;
    infiniopAddDescriptor_t bias_add_desc;
    size_t workspace_size;
    size_t weight_offset;
    size_t imediate_offset;
    size_t x1_cont_offset;
    size_t x2_cont_offset;
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

    // --- Weight Rearrangement ---
    // Target: [in1, out * in2] (flattened from [in1, out, in2])
    // Source: [out, in1, in2]
    
    // 1. Create descriptor for contiguous target weight [in1, out, in2]
    size_t weight_dst_shape[3] = {in1_features, out_features, in2_features};
    infiniopTensorDescriptor_t weight_dst_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&weight_dst_desc, 3, weight_dst_shape, nullptr, dtype));

    // 2. Create descriptor for source weight viewed as [in1, out, in2] (permuted)
    infiniopTensorDescriptor_t weight_src_permuted;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&weight_src_permuted, 3, weight_desc->shape().data(), weight_desc->strides().data(), dtype));
    TRANSFORM_TENSOR_DESC(weight_src_permuted, dimPermute({1, 0, 2}));

    // 3. Create Rearrange descriptor
    infiniopRearrangeDescriptor_t weight_rearrange_desc;
    CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &weight_rearrange_desc, weight_dst_desc, weight_src_permuted));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(weight_src_permuted));

    size_t weight_bytes = in1_features * out_features * in2_features * dtype_size;

    // --- GEMM 1: x1 @ weight_matrix ---
    // x1: [batch, in1]
    // weight_matrix: [in1, out * in2]
    // result (imediate): [batch, out * in2]

    // Prepare weight matrix descriptor (view of weight_dst_desc)
    infiniopTensorDescriptor_t weight_matrix_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&weight_matrix_desc, 3, weight_dst_shape, nullptr, dtype));
    TRANSFORM_TENSOR_DESC(weight_matrix_desc, dimMerge(1, 2));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(weight_dst_desc)); // Done with weight_dst_desc

    // Prepare x1 descriptor (handle non-contiguous)
    infiniopTensorDescriptor_t x1_gemm_desc = x1_desc;
    infiniopRearrangeDescriptor_t x1_rearrange_desc = nullptr;
    size_t x1_cont_bytes = 0;

    if (x1_desc->strides()[x1_desc->ndim() - 1] != 1) {
        infiniopTensorDescriptor_t x1_cont_desc;
        CHECK_STATUS(infiniopCreateTensorDescriptor(&x1_cont_desc, 2, x1_desc->shape().data(), nullptr, dtype));
        CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &x1_rearrange_desc, x1_cont_desc, x1_desc));
        x1_gemm_desc = x1_cont_desc; // Use the contiguous descriptor for GEMM creation
        x1_cont_bytes = batch_size * in1_features * dtype_size;
    }

    // Prepare imediate descriptor
    size_t imediate_shape[2] = {batch_size, out_features * in2_features};
    infiniopTensorDescriptor_t imediate_flat_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&imediate_flat_desc, 2, imediate_shape, nullptr, dtype));

    infiniopGemmDescriptor_t gemm_1_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &gemm_1_desc, imediate_flat_desc, x1_gemm_desc, weight_matrix_desc));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(weight_matrix_desc));
    if (x1_rearrange_desc) {
        CHECK_STATUS(infiniopDestroyTensorDescriptor(x1_gemm_desc));
    }

    size_t imediate_bytes = batch_size * out_features * in2_features * dtype_size;

    // --- GEMM 2: imediate @ x2^T ---
    // We perform Batch GEMM:
    // A (imediate): [batch, out, in2]
    // B (x2):       [batch, in2, 1]
    // C (out):      [batch, out, 1]

    // Prepare A: imediate viewed as [batch, out, in2]
    infiniopTensorDescriptor_t imediate_split_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&imediate_split_desc, 2, imediate_shape, nullptr, dtype));
    TRANSFORM_TENSOR_DESC(imediate_split_desc, dimSplit(1, {out_features, in2_features}));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(imediate_flat_desc)); // Done with flat

    // Prepare B: x2 viewed as [batch, in2, 1] (handle non-contiguous)
    infiniopTensorDescriptor_t x2_gemm_desc = x2_desc;
    infiniopRearrangeDescriptor_t x2_rearrange_desc = nullptr;
    size_t x2_cont_bytes = 0;

    if (x2_desc->strides()[x2_desc->ndim() - 1] != 1) {
        infiniopTensorDescriptor_t x2_cont_desc;
        CHECK_STATUS(infiniopCreateTensorDescriptor(&x2_cont_desc, 2, x2_desc->shape().data(), nullptr, dtype));
        CHECK_STATUS(infiniopCreateRearrangeDescriptor(handle, &x2_rearrange_desc, x2_cont_desc, x2_desc));
        x2_gemm_desc = x2_cont_desc;
        x2_cont_bytes = batch_size * in2_features * dtype_size;
    }

    infiniopTensorDescriptor_t x2_col_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&x2_col_desc, 2, x2_gemm_desc->shape().data(), x2_gemm_desc->strides().data(), dtype));
    TRANSFORM_TENSOR_DESC(x2_col_desc, dimSplit(1, {in2_features, 1}));
    if (x2_rearrange_desc) {
        CHECK_STATUS(infiniopDestroyTensorDescriptor(x2_gemm_desc));
    }

    // Prepare C: out viewed as [batch, out, 1]
    infiniopTensorDescriptor_t out_col_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&out_col_desc, 2, out_desc->shape().data(), out_desc->strides().data(), dtype));
    TRANSFORM_TENSOR_DESC(out_col_desc, dimSplit(1, {out_features, 1}));

    infiniopGemmDescriptor_t gemm_2_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &gemm_2_desc, out_col_desc, imediate_split_desc, x2_col_desc));
    
    CHECK_STATUS(infiniopDestroyTensorDescriptor(imediate_split_desc));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(x2_col_desc));
    CHECK_STATUS(infiniopDestroyTensorDescriptor(out_col_desc));

    // --- Bias Add ---
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

    // --- Workspace Calculation ---
    size_t gemm1_workspace_size = 0;
    size_t gemm2_workspace_size = 0;
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(gemm_1_desc, &gemm1_workspace_size));
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(gemm_2_desc, &gemm2_workspace_size));

    size_t op_workspace_size = std::max({gemm1_workspace_size, gemm2_workspace_size, add_workspace_size});
    op_workspace_size = aligned_size(op_workspace_size);

    size_t workspace_cursor = 0;
    auto reserve_buffer = [&](size_t bytes) -> size_t {
        if (bytes == 0) {
            return 0;
        }
        workspace_cursor = aligned_size(workspace_cursor);
        size_t offset = workspace_cursor;
        workspace_cursor += bytes;
        return offset;
    };

    size_t weight_offset = reserve_buffer(weight_bytes);
    size_t imediate_offset = reserve_buffer(imediate_bytes);
    size_t x1_cont_offset = x1_rearrange_desc ? reserve_buffer(x1_cont_bytes) : 0;
    size_t x2_cont_offset = x2_rearrange_desc ? reserve_buffer(x2_cont_bytes) : 0;
    size_t op_workspace_offset = reserve_buffer(op_workspace_size);
    size_t workspace_size = aligned_size(workspace_cursor);

    *(InfiniopBilinearDescriptor **)desc_ptr = new InfiniopBilinearDescriptor{
        {handle->device, handle->device_id},
        gemm_1_desc,
        gemm_2_desc,
        weight_rearrange_desc,
        x1_rearrange_desc,
        x2_rearrange_desc,
        bias_add_desc,
        workspace_size,
        weight_offset,
        imediate_offset,
        x1_cont_offset,
        x2_cont_offset,
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

    // 1. Rearrange Weight
    CHECK_STATUS(infiniopRearrange(desc->weight_rearrange_desc, weight_buffer, weight, stream));

    // 2. Prepare x1
    void const *x1_ptr = x1;
    if (desc->x1_rearrange_desc) {
        void *x1_buffer = workspace_ptr + desc->x1_cont_offset;
        CHECK_STATUS(infiniopRearrange(desc->x1_rearrange_desc, x1_buffer, x1, stream));
        x1_ptr = x1_buffer;
    }

    // 3. GEMM 1: x1 @ weight -> imediate
    CHECK_STATUS(infiniopGemm(desc->gemm_1_desc, op_workspace, desc->op_workspace_size,
                              imediate_buffer, x1_ptr, weight_buffer, 1.0f, 0.0f, stream));

    // 4. Prepare x2
    void const *x2_ptr = x2;
    if (desc->x2_rearrange_desc) {
        void *x2_buffer = workspace_ptr + desc->x2_cont_offset;
        CHECK_STATUS(infiniopRearrange(desc->x2_rearrange_desc, x2_buffer, x2, stream));
        x2_ptr = x2_buffer;
    }

    // 5. GEMM 2: imediate @ x2 -> out
    CHECK_STATUS(infiniopGemm(desc->gemm_2_desc, op_workspace, desc->op_workspace_size,
                              out, imediate_buffer, x2_ptr, 1.0f, 0.0f, stream));

    // 6. Bias Add
    if (desc->bias_add_desc) {
        CHECK_STATUS(infiniopAdd(desc->bias_add_desc, op_workspace, desc->op_workspace_size,
                                 out, out, bias, stream));
    }

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopDestroyBilinearDescriptor(
    infiniopBilinearDescriptor_t desc_) {
    auto desc = reinterpret_cast<InfiniopBilinearDescriptor *>(desc_);

    CHECK_STATUS(infiniopDestroyGemmDescriptor(desc->gemm_1_desc));
    CHECK_STATUS(infiniopDestroyGemmDescriptor(desc->gemm_2_desc));
    CHECK_STATUS(infiniopDestroyRearrangeDescriptor(desc->weight_rearrange_desc));
    if (desc->x1_rearrange_desc) {
        CHECK_STATUS(infiniopDestroyRearrangeDescriptor(desc->x1_rearrange_desc));
    }
    if (desc->x2_rearrange_desc) {
        CHECK_STATUS(infiniopDestroyRearrangeDescriptor(desc->x2_rearrange_desc));
    }
    if (desc->bias_add_desc) {
        CHECK_STATUS(infiniopDestroyAddDescriptor(desc->bias_add_desc));
    }

    delete desc;
    return INFINI_STATUS_SUCCESS;
}
