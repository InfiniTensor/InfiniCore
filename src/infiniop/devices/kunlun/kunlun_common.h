#ifndef __INFINIOP_KUNLUN_COMMON_H__
#define __INFINIOP_KUNLUN_COMMON_H__

// This header file will only be include by .xpu file
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_math.h"
#include "xpu/kernel/xtdk_simd.h"
#include "xpu/runtime.h"

// Get mask for kunlun xpu 512bit register calculation
// if data is not enough to 512bit, padding zero and use
// mask to identify real data
// 0 - i bit 1, others 0
inline __device__ float lowerBitMask(int i) {
    return (1 << (i + 1)) - 1;
}

#define INF 1e30f

#endif
