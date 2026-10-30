#include "gemm_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_gemm.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>

// Custom hash function for alpha beta pair<float, float>
struct FloatPairHash {
    size_t operator()(const std::pair<float, float> &p) const {
        uint64_t combined;
        std::memcpy(reinterpret_cast<char *>(&combined), &p.first, sizeof(float));
        std::memcpy(reinterpret_cast<char *>(&combined) + sizeof(float), &p.second, sizeof(float));

        return std::hash<uint64_t>()(combined);
    }
};

struct FloatPairEqual {
    bool operator()(const std::pair<float, float> &a, const std::pair<float, float> &b) const {
        return a.first == b.first && a.second == b.second;
    }
};

namespace op::gemm::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t c, a, b;
    // cubeMathType
    // see doc:
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnBatchMatMul.md
    int8_t mt;
    // whether B is handed to aclnnGemm as a contiguous [out, in] tensor with
    // transB=1 (see Descriptor::create); 0 = original behaviour (transB = 0).
    int8_t transB;
    // aclnnGemm accepts ND only. NZ weights use aclnnMatmulWeightNz and its
    // one-shot executor, which is consumed by the matching call API.
    int8_t weightNz;
    ~Opaque() {
        delete c;
        delete a;
        delete b;
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    return createWithFormat(
        handle_, desc_ptr, c_desc, a_desc, b_desc, false);
}

infiniStatus_t Descriptor::createWithFormat(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    bool b_is_fractal_nz) {
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::ROW_MAJOR);
    CHECK_RESULT(result);
    auto info = result.take();

    auto c = new aclnnTensorDescriptor(toAclDataType(c_desc->dtype()),
                                       {static_cast<int64_t>(info.m), static_cast<int64_t>(info.n)},
                                       {info.c_matrix.row_stride, info.c_matrix.col_stride});
    auto a = new aclnnTensorDescriptor(toAclDataType(a_desc->dtype()),
                                       {static_cast<int64_t>(info.a_matrix.rows), static_cast<int64_t>(info.a_matrix.cols)},
                                       {info.a_matrix.row_stride, info.a_matrix.col_stride});
    // The common Linear case stores the weight physically as [out, in] row-major
    // and views it here as a logical [in, out] column-major matrix (row_stride == 1).
    // Passing that column-major view to aclnnGemm with transB = 0 makes it emit a
    // physical Transpose kernel on every launch -- this dominated runtime (~60%).
    // Instead, describe the SAME memory as a contiguous [out, in] tensor and set
    // transB = 1, so the cube core handles the transpose via addressing (no kernel).
    // Mathematically identical; only affects Ascend. For any other B layout, keep
    // the original behaviour (transB = 0).
    bool trans_b = false;
    aclnnTensorDescriptor *b;
    if (b_is_fractal_nz) {
        CHECK_API_OR(info.batch == 1, true,
                     return INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_API_OR(
            info.b_matrix.row_stride == static_cast<ptrdiff_t>(info.b_matrix.cols) &&
                info.b_matrix.col_stride == 1,
            true, return INFINI_STATUS_BAD_TENSOR_STRIDES);
        const auto k = static_cast<int64_t>(info.b_matrix.rows);
        const auto n = static_cast<int64_t>(info.b_matrix.cols);
        CHECK_API_OR(k % 16 == 0 && n % 16 == 0, true,
                     return INFINI_STATUS_BAD_TENSOR_SHAPE);
        b = new aclnnTensorDescriptor(
            toAclDataType(b_desc->dtype()),
            {k, n}, {n, 1}, ACL_FORMAT_FRACTAL_NZ,
            {n / 16, k / 16, 16, 16});
    } else if (info.b_matrix.row_stride == 1 && info.b_matrix.col_stride > 1) {
        b = new aclnnTensorDescriptor(toAclDataType(b_desc->dtype()),
                                      {static_cast<int64_t>(info.b_matrix.cols), static_cast<int64_t>(info.b_matrix.rows)},
                                      {info.b_matrix.col_stride, info.b_matrix.row_stride});
        trans_b = true;
    } else {
        b = new aclnnTensorDescriptor(toAclDataType(b_desc->dtype()),
                                      {static_cast<int64_t>(info.b_matrix.rows), static_cast<int64_t>(info.b_matrix.cols)},
                                      {info.b_matrix.row_stride, info.b_matrix.col_stride});
    }

    auto tc = c->tensor,
         ta = a->tensor,
         tb = b->tensor;

    aclOpExecutor *executor = nullptr;
    size_t workspace_size = 0;
    int8_t mt = 1;
    if (b_is_fractal_nz) {
        CHECK_ACL(aclnnMatmulWeightNzGetWorkspaceSize(
            ta, tb, tc, mt, &workspace_size, &executor));
    } else {
        CHECK_ACL(aclnnGemmGetWorkspaceSize(ta, tb, tc, 1., 0., 0, trans_b ? 1 : 0, tc, mt, &workspace_size, &executor));
        size_t beta_one_workspace_size = 0;
        aclOpExecutor *beta_one_executor = nullptr;
        CHECK_ACL(aclnnGemmGetWorkspaceSize(ta, tb, tc, 1., 1., 0, trans_b ? 1 : 0, tc, mt, &beta_one_workspace_size, &beta_one_executor));
        workspace_size = std::max(workspace_size, beta_one_workspace_size);
    }

    *desc_ptr = new Descriptor(
        dtype, info, workspace_size,
        new Opaque{
            c,
            a,
            b,
            mt,
            static_cast<int8_t>(trans_b ? 1 : 0),
            static_cast<int8_t>(b_is_fractal_nz ? 1 : 0)},
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspaceSize_,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    size_t workspace_size = _workspace_size;
    if (_opaque->weightNz) {
        // The dedicated WeightNz API has matmul semantics only. Callers must
        // keep bias/scaling on the ND path unless these values are identity.
        if (alpha != 1.0f || beta != 0.0f || _info.batch != 1) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }

        aclnnTensorDescriptor call_a(
            _opaque->a->dataType, _opaque->a->shape, _opaque->a->strides,
            _opaque->a->format, _opaque->a->storageShape,
            const_cast<void *>(a));
        aclnnTensorDescriptor call_b(
            _opaque->b->dataType, _opaque->b->shape, _opaque->b->strides,
            _opaque->b->format, _opaque->b->storageShape,
            const_cast<void *>(b));
        aclnnTensorDescriptor call_c(
            _opaque->c->dataType, _opaque->c->shape, _opaque->c->strides,
            _opaque->c->format, _opaque->c->storageShape, c);

        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnMatmulWeightNzGetWorkspaceSize(
            call_a.tensor, call_b.tensor, call_c.tensor, _opaque->mt,
            &workspace_size, &executor));
        if (workspaceSize_ < workspace_size) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }

        CHECK_ACL(aclnnMatmulWeightNz(
            workspace, workspace_size, executor,
            reinterpret_cast<aclrtStream>(stream)));
        return INFINI_STATUS_SUCCESS;
    }

    auto unit = infiniSizeOf(_dtype);
    for (size_t i = 0; i < _info.batch; ++i) {
        auto a_ptr = ((char *)const_cast<void *>(a))
                   + i * _info.a_matrix.stride * unit;
        auto b_ptr = ((char *)const_cast<void *>(b))
                   + i * _info.b_matrix.stride * unit;
        auto c_ptr = ((char *)c) + i * _info.c_matrix.stride * unit;
        aclnnTensorDescriptor call_a(
            _opaque->a->dataType, _opaque->a->shape, _opaque->a->strides,
            _opaque->a->format, _opaque->a->storageShape, a_ptr);
        aclnnTensorDescriptor call_b(
            _opaque->b->dataType, _opaque->b->shape, _opaque->b->strides,
            _opaque->b->format, _opaque->b->storageShape, b_ptr);
        aclnnTensorDescriptor call_c(
            _opaque->c->dataType, _opaque->c->shape, _opaque->c->strides,
            _opaque->c->format, _opaque->c->storageShape, c_ptr);

        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnGemmGetWorkspaceSize(
            call_a.tensor, call_b.tensor, call_c.tensor,
            alpha, beta, 0, _opaque->transB, call_c.tensor, _opaque->mt,
            &workspace_size, &executor));
        if (workspaceSize_ < workspace_size) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        CHECK_ACL(aclnnGemm(workspace, workspace_size, executor, stream));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::ascend
