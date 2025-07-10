#ifndef __RELU_CUDA_API_H__
#define __RELU_CUDA_API_H__

#ifdef ENABLE_NINETOOTHED

#include "../../../elementwise/cuda/elementwise_cuda_api.cuh"

ELEMENTWISE_DESCRIPTOR(relu, cuda)

#endif

#endif // __RELU_CUDA_API_H__
