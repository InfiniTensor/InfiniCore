#ifndef __INFINIOP_UNARY_OPS_API_H__
#define __INFINIOP_UNARY_OPS_API_H__

#include "unary_op_api.h"

/**
 * @brief Unified API declarations for all unary operators.
 * 
 * This header contains API declarations for all unary operators in a single file,
 * eliminating the need for individual header files for each operator.
 * 
 * All unary operator APIs are declared here:
 * - abs, log, sqrt, reciprocal, neg, round, sinh, sign, tan
 * - acosh, asinh, cos, atanh, asin, floor, cosh, erf, atan, acos, ceil
 */

// Declare all unary operator APIs
UNARY_OP_API_DECLARE(abs, Abs)
UNARY_OP_API_DECLARE(log, Log)
UNARY_OP_API_DECLARE(log2, Log2)
UNARY_OP_API_DECLARE(log10, Log10)
UNARY_OP_API_DECLARE(log1p, Log1p)
UNARY_OP_API_DECLARE(sqrt, Sqrt)
UNARY_OP_API_DECLARE(square, Square)
UNARY_OP_API_DECLARE(rsqrt, Rsqrt)
UNARY_OP_API_DECLARE(reciprocal, Reciprocal)
UNARY_OP_API_DECLARE(neg, Neg)
UNARY_OP_API_DECLARE(round, Round)
UNARY_OP_API_DECLARE(sinh, Sinh)
UNARY_OP_API_DECLARE(sign, Sign)
UNARY_OP_API_DECLARE(tan, Tan)
UNARY_OP_API_DECLARE(acosh, Acosh)
UNARY_OP_API_DECLARE(asinh, Asinh)
UNARY_OP_API_DECLARE(cos, Cos)
UNARY_OP_API_DECLARE(atanh, Atanh)
UNARY_OP_API_DECLARE(asin, Asin)
UNARY_OP_API_DECLARE(floor, Floor)
UNARY_OP_API_DECLARE(cosh, Cosh)
UNARY_OP_API_DECLARE(erf, Erf)
UNARY_OP_API_DECLARE(atan, Atan)
UNARY_OP_API_DECLARE(acos, Acos)
UNARY_OP_API_DECLARE(ceil, Ceil)
UNARY_OP_API_DECLARE(exp, Exp)
UNARY_OP_API_DECLARE(exp2, Exp2)
UNARY_OP_API_DECLARE(hardswish, Hardswish)
UNARY_OP_API_DECLARE(isnan, IsNan)
UNARY_OP_API_DECLARE(isinf, IsInf)
UNARY_OP_API_DECLARE(isfinite, IsFinite)
UNARY_OP_API_DECLARE(sinc, Sinc)
UNARY_OP_API_DECLARE(sin, Sin)

#endif // __INFINIOP_UNARY_OPS_API_H__
