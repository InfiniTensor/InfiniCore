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

} // namespace op::gemm::kunlun

#endif // __GEMM_KUNLUN_CAST_H__
