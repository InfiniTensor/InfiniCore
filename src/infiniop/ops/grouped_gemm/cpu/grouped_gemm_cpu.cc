#include "grouped_gemm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <cstdint>

namespace op::grouped_gemm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t group_sizes_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    CHECK_DTYPE(c_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = GroupedGemmInfo::create(c_desc, a_desc, b_desc, group_sizes_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(), 0,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
static void calculate_typed(
    const GroupedGemmInfo &info,
    void *c_,
    const void *a_,
    const void *b_,
    const int32_t *group_sizes,
    float alpha,
    float beta) {

    auto a = reinterpret_cast<const Tdata *>(a_);
    auto b = reinterpret_cast<const Tdata *>(b_);
    auto c = reinterpret_cast<Tdata *>(c_);

    // Walk each group and dispatch a plain triple-loop matmul on its rows.
    // Parallelism is across groups + rows; the tight K loop stays sequential.
    ptrdiff_t row_offset = 0;
    for (size_t g = 0; g < info.num_groups; ++g) {
        ptrdiff_t rows = group_sizes[g];
        if (rows <= 0) {
            // Skip degenerate / empty groups; routing is allowed to produce them.
            continue;
        }
        const Tdata *b_g = b + g * info.b_group_stride;

#pragma omp parallel for collapse(2)
        for (ptrdiff_t r = 0; r < rows; ++r) {
            for (ptrdiff_t n = 0; n < ptrdiff_t(info.n); ++n) {
                const Tdata *a_row = a + (row_offset + r) * info.a_row_stride;
                const Tdata *b_row = b_g + n * info.b_row_stride;
                Tdata *c_pos = c + (row_offset + r) * info.c_row_stride + n * info.c_col_stride;

                float sum = 0.f;
                for (ptrdiff_t k = 0; k < ptrdiff_t(info.k); ++k) {
                    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                        sum += utils::cast<float>(a_row[k * info.a_col_stride])
                               * utils::cast<float>(b_row[k * info.b_col_stride]);
                    } else {
                        sum += a_row[k * info.a_col_stride] * b_row[k * info.b_col_stride];
                    }
                }

                if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                    if (beta == 0.f) {
                        *c_pos = utils::cast<Tdata>(alpha * sum);
                    } else {
                        *c_pos = utils::cast<Tdata>(beta * utils::cast<float>(*c_pos) + alpha * sum);
                    }
                } else {
                    if (beta == 0.f) {
                        *c_pos = alpha * sum;
                    } else {
                        *c_pos = beta * (*c_pos) + alpha * sum;
                    }
                }
            }
        }
        row_offset += rows;
    }
}

infiniStatus_t Descriptor::calculate(
    void * /*workspace*/, size_t /*workspace_size*/,
    void *c,
    const void *a,
    const void *b,
    const void *group_sizes,
    const void *group_sizes_host,
    float alpha,
    float beta,
    void * /*stream*/) const {

    // On CPU `group_sizes` is already host memory, so the optional host copy is
    // redundant; prefer it when supplied to mirror the device backends' contract.
    auto sizes = reinterpret_cast<const int32_t *>(
        group_sizes_host != nullptr ? group_sizes_host : group_sizes);

    // Validate that group_sizes sums to m_total to keep `row_offset + r` in range.
    {
        size_t total = 0;
        for (size_t g = 0; g < _info.num_groups; ++g) {
            if (sizes[g] < 0) {
                return INFINI_STATUS_BAD_PARAM;
            }
            total += size_t(sizes[g]);
        }
        if (total != _info.m_total) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        calculate_typed<fp16_t>(_info, c, a, b, sizes, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        calculate_typed<bf16_t>(_info, c, a, b, sizes, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        calculate_typed<float>(_info, c, a, b, sizes, alpha, beta);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::grouped_gemm::cpu
