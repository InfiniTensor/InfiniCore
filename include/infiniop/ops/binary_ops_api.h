#ifndef __INFINIOP_BINARY_OPS_API_H__
#define __INFINIOP_BINARY_OPS_API_H__

#include "binary_op_api.h"

/**
 * @brief Unified API declarations for all binary operators.
 * 
 * This header contains API declarations for all binary operators in a single file,
 * eliminating the need for individual header files for each operator.
 * 
 * All binary operator APIs are declared here:
 * - div, pow, mod, max, min
 */

// Declare all binary operator APIs
BINARY_OP_API_DECLARE(div, Div)
BINARY_OP_API_DECLARE(floor_divide, FloorDivide)
BINARY_OP_API_DECLARE(pow, Pow)
BINARY_OP_API_DECLARE(copysign, CopySign)
BINARY_OP_API_DECLARE(hypot, Hypot)
BINARY_OP_API_DECLARE(atan2, Atan2)
BINARY_OP_API_DECLARE(mod, Mod)
BINARY_OP_API_DECLARE(remainder, Remainder)
BINARY_OP_API_DECLARE(max, Max)
BINARY_OP_API_DECLARE(min, Min)
BINARY_OP_API_DECLARE(fmax, Fmax)
BINARY_OP_API_DECLARE(fmin, Fmin)
BINARY_OP_API_DECLARE(gt, Gt)
BINARY_OP_API_DECLARE(lt, Lt)
BINARY_OP_API_DECLARE(ge, Ge)
BINARY_OP_API_DECLARE(le, Le)
BINARY_OP_API_DECLARE(eq, Eq)
BINARY_OP_API_DECLARE(ne, Ne)
BINARY_OP_API_DECLARE(logical_and, LogicalAnd)
BINARY_OP_API_DECLARE(logical_or, LogicalOr)
BINARY_OP_API_DECLARE(logical_xor, LogicalXor)
BINARY_OP_API_DECLARE(bitwise_and, BitwiseAnd)
BINARY_OP_API_DECLARE(bitwise_or, BitwiseOr)
BINARY_OP_API_DECLARE(bitwise_xor, BitwiseXor)
BINARY_OP_API_DECLARE(bitwise_left_shift, BitwiseLeftShift)
BINARY_OP_API_DECLARE(bitwise_right_shift, BitwiseRightShift)

#endif // __INFINIOP_BINARY_OPS_API_H__
