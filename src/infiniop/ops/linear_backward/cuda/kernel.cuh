#ifndef __LINEAR_BACKWARD_KERNEL_CUH__
#define __LINEAR_BACKWARD_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../../utils/custom_types.h"
#include "../../../tensor.h"
#include <cuda_bf16.h>

namespace op::linear_backward::cuda {

// Linear backward CUDA kernel
// Computes gradients for linear layer:
// grad_x = grad_y * w^T
// grad_w = grad_y^T * x  
// grad_b = sum(grad_y, dim=0)
//
// Parameters:
// - grad_x: gradient w.r.t. input [batch_size, in_features]
// - grad_w: gradient w.r.t. weight [out_features, in_features]
// - grad_b: gradient w.r.t. bias [out_features]
// - grad_y: gradient w.r.t. output [batch_size, out_features]
// - x: input tensor [batch_size, in_features]
// - w: weight tensor [out_features, in_features]
// - x_shape: shape of x tensor
// - w_shape: shape of w tensor
// - x_strides: strides of x tensor
// - w_strides: strides of w tensor
// - grad_y_strides: strides of grad_y tensor
// - grad_x_strides: strides of grad_x tensor (if not null)
// - grad_w_strides: strides of grad_w tensor (if not null)
// - grad_b_strides: strides of grad_b tensor (if not null)
__global__ void linear_backward_kernel(
    void* grad_x,
    void* grad_w, 
    void* grad_b,
    const void* grad_y,
    const void* x,
    const void* w,
    int batch_size,
    int in_features,
    int out_features,
    infiniDtype_t dtype) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dtype == INFINI_DTYPE_F16) {
        half* grad_x_ptr = (half*)grad_x;
        half* grad_w_ptr = (half*)grad_w;
        half* grad_b_ptr = (half*)grad_b;
        const half* grad_y_ptr = (const half*)grad_y;
        const half* x_ptr = (const half*)x;
        const half* w_ptr = (const half*)w;
        
        // Compute grad_x = grad_y @ w^T (grad_y * w)
        if (grad_x_ptr && idx < batch_size * in_features) {
            int batch = idx / in_features;
            int in = idx % in_features;
            
            float sum = 0.0f;
            for (int out = 0; out < out_features; out++) {
                // w is stored as [out_features, in_features], so w^T[in][out] = w[out][in]
                sum += __half2float(grad_y_ptr[batch * out_features + out]) * 
                       __half2float(w_ptr[out * in_features + in]);
            }
            grad_x_ptr[idx] = __float2half(sum);
        }
        
        // Compute grad_w = grad_y^T @ x
        if (grad_w_ptr && idx < out_features * in_features) {
            int out = idx / in_features;
            int in = idx % in_features;
            
            float sum = 0.0f;
            for (int batch = 0; batch < batch_size; batch++) {
                sum += __half2float(grad_y_ptr[batch * out_features + out]) * 
                       __half2float(x_ptr[batch * in_features + in]);
            }
            grad_w_ptr[idx] = __float2half(sum);
        }
        
        // Compute grad_b = sum(grad_y, dim=0)
        if (grad_b_ptr && idx < out_features) {
            float sum = 0.0f;
            for (int batch = 0; batch < batch_size; batch++) {
                sum += __half2float(grad_y_ptr[batch * out_features + idx]);
            }
            grad_b_ptr[idx] = __float2half(sum);
        }
    }
    else if (dtype == INFINI_DTYPE_F32) {
        float* grad_x_ptr = (float*)grad_x;
        float* grad_w_ptr = (float*)grad_w;
        float* grad_b_ptr = (float*)grad_b;
        const float* grad_y_ptr = (const float*)grad_y;
        const float* x_ptr = (const float*)x;
        const float* w_ptr = (const float*)w;
        
        // Compute grad_x = grad_y @ w^T (grad_y * w)
        if (grad_x_ptr && idx < batch_size * in_features) {
            int batch = idx / in_features;
            int in = idx % in_features;
            
            float sum = 0.0f;
            for (int out = 0; out < out_features; out++) {
                // w is stored as [out_features, in_features], so w^T[in][out] = w[out][in]
                sum += grad_y_ptr[batch * out_features + out] * 
                       w_ptr[out * in_features + in];
            }
            grad_x_ptr[idx] = sum;
        }
        
        // Compute grad_w = grad_y^T @ x
        if (grad_w_ptr && idx < out_features * in_features) {
            int out = idx / in_features;
            int in = idx % in_features;
            
            float sum = 0.0f;
            for (int batch = 0; batch < batch_size; batch++) {
                sum += grad_y_ptr[batch * out_features + out] * 
                       x_ptr[batch * in_features + in];
            }
            grad_w_ptr[idx] = sum;
        }
        
        // Compute grad_b = sum(grad_y, dim=0)
        if (grad_b_ptr && idx < out_features) {
            float sum = 0.0f;
            for (int batch = 0; batch < batch_size; batch++) {
                sum += grad_y_ptr[batch * out_features + idx];
            }
            grad_b_ptr[idx] = sum;
         }
     }
     else if (dtype == INFINI_DTYPE_BF16) {
         __nv_bfloat16* grad_x_ptr = (__nv_bfloat16*)grad_x;
         __nv_bfloat16* grad_w_ptr = (__nv_bfloat16*)grad_w;
         __nv_bfloat16* grad_b_ptr = (__nv_bfloat16*)grad_b;
         const __nv_bfloat16* grad_y_ptr = (const __nv_bfloat16*)grad_y;
         const __nv_bfloat16* x_ptr = (const __nv_bfloat16*)x;
         const __nv_bfloat16* w_ptr = (const __nv_bfloat16*)w;
         
         // Compute grad_x = grad_y @ w^T (grad_y * w)
         if (grad_x_ptr && idx < batch_size * in_features) {
             int batch = idx / in_features;
             int in = idx % in_features;
             
             float sum = 0.0f;
             for (int out = 0; out < out_features; out++) {
                 // w is stored as [out_features, in_features], so w^T[in][out] = w[out][in]
                 sum += __bfloat162float(grad_y_ptr[batch * out_features + out]) * 
                        __bfloat162float(w_ptr[out * in_features + in]);
             }
             grad_x_ptr[idx] = __float2bfloat16(sum);
         }
         
         // Compute grad_w = grad_y^T @ x
         if (grad_w_ptr && idx < out_features * in_features) {
             int out = idx / in_features;
             int in = idx % in_features;
             
             float sum = 0.0f;
             for (int batch = 0; batch < batch_size; batch++) {
                 sum += __bfloat162float(grad_y_ptr[batch * out_features + out]) * 
                        __bfloat162float(x_ptr[batch * in_features + in]);
             }
             grad_w_ptr[idx] = __float2bfloat16(sum);
         }
         
         // Compute grad_b = sum(grad_y, dim=0)
         if (grad_b_ptr && idx < out_features) {
             float sum = 0.0f;
             for (int batch = 0; batch < batch_size; batch++) {
                 sum += __bfloat162float(grad_y_ptr[batch * out_features + idx]);
             }
             grad_b_ptr[idx] = __float2bfloat16(sum);
         }
     }
}

} // namespace op::linear_backward::cuda

#endif // __LINEAR_BACKWARD_KERNEL_CUH__