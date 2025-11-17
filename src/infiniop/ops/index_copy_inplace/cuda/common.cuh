#ifndef __INFINIOP_INDEX_COPY_INPLACE_CUDA_COMMON_CUH__
#define __INFINIOP_INDEX_COPY_INPLACE_CUDA_COMMON_CUH__
//.cu文件导入/#include "../../../devices/nvidia/nvidia_kernel_common.cuh"会报错
//所以就自己把用到的部分单独拿出来

// 用包含 CUDA 官方的头文件来定义 bfloat16 和 half
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using cuda_bfloat16 = nv_bfloat16;

#endif // __INFINIOP_INDEX_COPY_INPLACE_CUDA_COMMON_CUH__