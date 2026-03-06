#ifndef __INFINIOP_UTILS_H__
#define __INFINIOP_UTILS_H__

// InfiniOp internal utility umbrella header.
// Most operator implementations include this header via a relative path like "../../../utils.h".
// It provides:
// - common dtype/shape/status check macros (CHECK_*)
// - utils::Result and CHECK_RESULT
// - base utility helpers from src/utils.h

#include "../utils/result.hpp"
#include "tensor.h"

#endif // __INFINIOP_UTILS_H__
