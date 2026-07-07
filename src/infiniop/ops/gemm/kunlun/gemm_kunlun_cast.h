#ifndef __GEMM_KUNLUN_CAST_H__
#define __GEMM_KUNLUN_CAST_H__

#include "../../../devices/kunlun/kunlun_common.h"
#include "../info.h"

namespace op::gemm::kunlun {

infiniStatus_t castBf16ToF16(
    const void *src,
    void *dst,
    const BlasMatrix &matrix,
    size_t batch,
    kunlunStream_t stream);

infiniStatus_t castF16ToBf16(
    const void *src,
    void *dst,
    const BlasMatrix &matrix,
    size_t batch,
    kunlunStream_t stream);

infiniStatus_t castBf16ToF32(
    const void *src,
    void *dst,
    const BlasMatrix &matrix,
    size_t batch,
    kunlunStream_t stream);

infiniStatus_t castF32ToBf16(
    const void *src,
    void *dst,
    const BlasMatrix &matrix,
    size_t batch,
    kunlunStream_t stream);

infiniStatus_t castBf16ToF32Chunk(
    const void *src,
    void *dst,
    const BlasMatrix &matrix,
    size_t batch,
    size_t row_offset,
    size_t rows,
    kunlunStream_t stream);

infiniStatus_t castF32ToBf16Chunk(
    const void *src,
    void *dst,
    const BlasMatrix &matrix,
    size_t batch,
    size_t row_offset,
    size_t rows,
    kunlunStream_t stream);

infiniStatus_t zeroPackedBuffer(
    void *dst,
    size_t bytes,
    kunlunStream_t stream);

infiniStatus_t bf16SkinnyGemv(
    const void *a,
    const void *b,
    void *c,
    const BlasMatrix &a_matrix,
    const BlasMatrix &b_matrix,
    const BlasMatrix &c_matrix,
    size_t m,
    size_t k,
    float alpha,
    float beta,
    kunlunStream_t stream);

} // namespace op::gemm::kunlun

#endif // __GEMM_KUNLUN_CAST_H__
