#ifndef __AWQ_MARLIN_REPACK_INFO_H__
#define __AWQ_MARLIN_REPACK_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "marlin/marlin.cuh"
#include <vector>

#include <cassert>

namespace op::awq_marlin_repack {

class AwqMarlinRepackInfo {
    AwqMarlinRepackInfo() = default;

public:
    infiniDtype_t output_dtype, input_dtype;
    size_t size_k, size_n;
    int64_t num_bits;
    bool is_a_8bit;

    static utils::Result<AwqMarlinRepackInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        int64_t num_bits,
        bool is_a_8bit) {
        CHECK_OR_RETURN(
            output_desc != nullptr && input_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);
        const infiniDtype_t output_dtype = output_desc->dtype();
        const infiniDtype_t input_dtype = input_desc->dtype();
        CHECK_DTYPE(input_dtype, INFINI_DTYPE_I32);
        CHECK_DTYPE(input_dtype, output_dtype);

        size_t size_k = input_desc->dim(0);
        int const pack_factor = 32 / num_bits;
        size_t size_n = input_desc->dim(1) * pack_factor;

        CHECK_OR_RETURN(size_k / marlin::tile_size == output_desc->dim(0) || size_n * marlin::tile_size / pack_factor == output_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<AwqMarlinRepackInfo>(
            AwqMarlinRepackInfo{output_dtype, input_dtype, size_k, size_n, num_bits, is_a_8bit});
    }
};

} // namespace op::awq_marlin_repack

#endif // __AWQ_MARLIN_REPACK_INFO_H__
