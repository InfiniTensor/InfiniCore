#pragma once

#include "attention_generic.cuh"
#include "attention_utils.cuh"
#include "cuda_compat.h"
#include "dtype_bfloat16.cuh"
#include "dtype_float32.cuh"
#include "dtype_fp8.cuh"

#if defined(ENABLE_NVIDIA_API)
#include "../nvidia/dtype_bfloat16.cuh"
#include "../nvidia/dtype_float16.cuh"

#endif

#if defined(ENABLE_METAX_API)
#include "../metax/dtype_bfloat16.cuh"
#include "../metax/dtype_float16.cuh"
#endif
