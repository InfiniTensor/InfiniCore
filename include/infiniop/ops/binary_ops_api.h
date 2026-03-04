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
BINARY_OP_API_DECLARE(pow, Pow)
BINARY_OP_API_DECLARE(mod, Mod)
BINARY_OP_API_DECLARE(max, Max)
BINARY_OP_API_DECLARE(min, Min)

#endif // __INFINIOP_BINARY_OPS_API_H__
