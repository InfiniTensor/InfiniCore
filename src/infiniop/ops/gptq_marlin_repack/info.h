#ifndef __AWQ_MARLIN_REPACK_INFO_H__
#define __AWQ_MARLIN_REPACK_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "marlin/marlin.cuh"
#include <vector>

#include <cassert>

namespace op::gptq_marlin_repack {

class GptqMarlinRepackInfo {
    GptqMarlinRepackInfo() = default;

public:
    infiniDtype_t output_dtype, input_dtype;
    size_t size_k, size_n;
    int64_t num_bits;
    bool is_a_8bit, has_perm;

    static utils::Result<GptqMarlinRepackInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t perm_desc,
        int64_t num_bits,
        bool is_a_8bit) {
        CHECK_OR_RETURN(
            output_desc != nullptr && input_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);
        const infiniDtype_t output_dtype = output_desc->dtype();
        const infiniDtype_t input_dtype = input_desc->dtype();

        CHECK_DTYPE(input_dtype, output_dtype);

        int const pack_factor = 32 / num_bits;
        size_t size_k = input_desc->dim(0) * pack_factor;
        size_t size_n = input_desc->dim(1);

        CHECK_OR_RETURN(size_k / marlin::tile_size == output_desc->dim(0) || size_n * marlin::tile_size / pack_factor == output_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        bool has_perm = false;

        if (perm_desc != nullptr && perm_desc->dim(0) != 0) {
            has_perm = true;
        }

        return utils::Result<GptqMarlinRepackInfo>(
            GptqMarlinRepackInfo{output_dtype, input_dtype, size_k, size_n, num_bits, is_a_8bit, has_perm});
    }
};

} // namespace op::gptq_marlin_repack

#endif // __AWQ_MARLIN_REPACK_INFO_H__
