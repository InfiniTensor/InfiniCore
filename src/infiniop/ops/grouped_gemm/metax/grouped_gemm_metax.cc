#include "grouped_gemm_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

#define CHECK_METAX(API) CHECK_INTERNAL(API, hcSuccess)

#include <cstdint>
#include <vector>

namespace op::grouped_gemm::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t group_sizes_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);

    CHECK_DTYPE(c_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = GroupedGemmInfo::create(c_desc, a_desc, b_desc, group_sizes_desc);
    CHECK_RESULT(result);

    auto info = result.take();

    *desc_ptr = new Descriptor(
        info, 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

struct McblasDtypes {
    hpccDataType ab_type;
    hpccDataType c_type;
    hcblasComputeType_t compute_type;
};

infiniStatus_t resolveDtypes(infiniDtype_t dtype, McblasDtypes &out) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        out.ab_type = out.c_type = HPCC_R_16F;
        out.compute_type = HCBLAS_COMPUTE_32F;
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        out.ab_type = out.c_type = HPCC_R_16BF;
        out.compute_type = HCBLAS_COMPUTE_32F;
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        out.ab_type = out.c_type = HPCC_R_32F;
        out.compute_type = HCBLAS_COMPUTE_32F_FAST_TF32;
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

inline size_t element_size(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
    case INFINI_DTYPE_BF16:
        return 2;
    case INFINI_DTYPE_F32:
        return 4;
    default:
        return 0;
    }
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    const void *group_sizes,
    const void *group_sizes_host,
    float alpha,
    float beta,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    McblasDtypes dtypes;
    CHECK_STATUS(resolveDtypes(_info.dtype, dtypes));

    auto hc_stream = static_cast<hcStream_t>(stream);

    // Resolve the host-side group sizes. When the caller already holds them on
    // the host (MoE routing is bucketed on CPU) it passes `group_sizes_host`,
    // letting us skip the per-call device->host copy + stream sync. On MetaX a
    // pageable hcMemcpyAsync is effectively blocking, so eliminating this
    // round-trip removes a hard sync from every grouped GEMM on the decode path.
    std::vector<int32_t> sizes_host_vec;
    const int32_t *sizes_host;
    if (group_sizes_host != nullptr) {
        sizes_host = reinterpret_cast<const int32_t *>(group_sizes_host);
    } else {
        sizes_host_vec.resize(_info.num_groups);
        CHECK_METAX(hcMemcpyAsync(
            sizes_host_vec.data(), group_sizes,
            _info.num_groups * sizeof(int32_t),
            hcMemcpyDeviceToHost, hc_stream));
        CHECK_METAX(hcStreamSynchronize(hc_stream));
        sizes_host = sizes_host_vec.data();
    }

    const size_t elem_size = element_size(_info.dtype);
    auto a_bytes = reinterpret_cast<const uint8_t *>(a);
    auto b_bytes = reinterpret_cast<const uint8_t *>(b);
    auto c_bytes = reinterpret_cast<uint8_t *>(c);

    // Row-major C[rows_g, N] = alpha * A[rows_g, K] @ B[g][N, K]^T + beta * C.
    // We compute the transpose `C^T = B @ A^T` in column-major space; the
    // memory is identical. See NVIDIA backend for the full derivation.
    CHECK_STATUS(_opaque->internal->useMcblas(
        hc_stream,
        [&](hcblasHandle_t handle) {
            ptrdiff_t row_offset = 0;
            for (size_t g = 0; g < _info.num_groups; ++g) {
                int32_t rows = sizes_host[g];
                if (rows <= 0) {
                    continue;
                }
                const void *a_g = a_bytes + size_t(row_offset) * size_t(_info.a_row_stride) * elem_size;
                const void *b_g = b_bytes + g * size_t(_info.b_group_stride) * elem_size;
                void *c_g = c_bytes + size_t(row_offset) * size_t(_info.c_row_stride) * elem_size;

                CHECK_MCBLAS(hcblasGemmEx(
                    handle,
                    HCBLAS_OP_T, // op on B[g] view -> N x K
                    HCBLAS_OP_N, // op on A view -> K x rows_g
                    static_cast<int>(_info.n),   // m_cb
                    static_cast<int>(rows),      // n_cb
                    static_cast<int>(_info.k),   // k_cb
                    &alpha,
                    b_g, dtypes.ab_type,
                    static_cast<int>(_info.b_row_stride),
                    a_g, dtypes.ab_type,
                    static_cast<int>(_info.a_row_stride),
                    &beta,
                    c_g, dtypes.c_type,
                    static_cast<int>(_info.c_row_stride),
                    dtypes.compute_type,
                    HCBLAS_GEMM_DEFAULT_TENSOR_OP));

                row_offset += rows;
            }
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::grouped_gemm::metax
