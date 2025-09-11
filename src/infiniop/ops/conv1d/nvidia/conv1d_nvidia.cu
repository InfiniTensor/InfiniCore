#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include <cudnn.h>
#include "conv1d_nvidia.cuh"
#include "infiniop/ops/conv1d.h"
#include "../../../tensor.h"
#include "../info.h"

#include <cuda_fp16.h>
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif
#endif

#define DESTROY_CUDNN_DESCRIPTOR(ptr, destroy) \
    do { if (ptr) { destroy(ptr); ptr = nullptr; } } while (0)

namespace op::conv1d::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    size_t workspace_size = 0;     // total workspace required
    size_t cudnn_ws_size = 0;      // cuDNN internal workspace size
    size_t conv_out_bytes = 0;     // extra bytes for gated temp output

#ifdef ENABLE_CUDNN_API
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;   // conv output descriptor (may be 2*Cout when gated)
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;   // bias for conv output (matches y_desc C)
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
#endif

    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> p) 
        : internal(std::move(p)) {}

    ~Opaque() {
#ifdef ENABLE_CUDNN_API
        DESTROY_CUDNN_DESCRIPTOR(x_desc, cudnnDestroyTensorDescriptor);
        DESTROY_CUDNN_DESCRIPTOR(y_desc, cudnnDestroyTensorDescriptor);
        DESTROY_CUDNN_DESCRIPTOR(w_desc, cudnnDestroyFilterDescriptor);
        DESTROY_CUDNN_DESCRIPTOR(b_desc, cudnnDestroyTensorDescriptor);
        DESTROY_CUDNN_DESCRIPTOR(conv_desc, cudnnDestroyConvolutionDescriptor);
#endif
    }
};

Descriptor::~Descriptor() { if (_opaque) delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t w,
    infiniopTensorDescriptor_t b,
    const void *pads_,
    const void *strides_,
    const void *dilations_,
    size_t n)
{
    if (n != 1) return INFINI_STATUS_NOT_IMPLEMENTED; // only support 1d

    auto h = reinterpret_cast<device::nvidia::Handle*>(handle_);
    auto dtype = y->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // Create Conv1dInfo to analyze the operation
    auto info_result = Conv1dInfo::create(handle_, y, x, w, b, pads_, strides_, dilations_, n);
    if (!info_result) {
        return info_result.status();
    }
    auto info = info_result.take();

    auto opaque = new Opaque(h->internal());

#ifdef ENABLE_CUDNN_API
    return h->internal()->useCudnn(nullptr, [&](cudnnHandle_t cudnn) {
        // Initialize cuDNN descriptors for 1D convolution
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&opaque->x_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&opaque->y_desc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&opaque->w_desc));
        if (b) CHECK_CUDNN(cudnnCreateTensorDescriptor(&opaque->b_desc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&opaque->conv_desc));
        
        // Set tensor descriptors for 1D conv (treat as 2D with height=1)
        const int B = static_cast<int>(info.batch());
        const int Cin = static_cast<int>(info.in_channels());
        const int Lin = static_cast<int>(info.input_dim(0));
        const int Lout = static_cast<int>(info.output_dim(0));
        const int groups = static_cast<int>(info.groups());
        const int OC_raw = static_cast<int>(info.gated() ? info.out_channels() * 2 : info.out_channels());
        const int Cin_per_group = Cin / groups;

        // x: [B, Cin, 1, Lin]
        int x_dims[] = {B, Cin, 1, Lin};
        int x_strides[] = {Cin * 1 * Lin, 1 * Lin, Lin, 1};

        // y_conv: [B, OC_raw, 1, Lout]
        int y_conv_dims[] = {B, OC_raw, 1, Lout};
        int y_conv_strides[] = {OC_raw * 1 * Lout, 1 * Lout, Lout, 1};

        // w: [OC_raw, Cin_per_group, 1, K]
        int w_dims[] = {OC_raw, Cin_per_group, 1, (int)info.kernel_dim(0)};
        
        cudnnDataType_t cudnn_dtype;
        if (dtype == INFINI_DTYPE_F32) cudnn_dtype = CUDNN_DATA_FLOAT;
        else if (dtype == INFINI_DTYPE_F16) cudnn_dtype = CUDNN_DATA_HALF;
        else if (dtype == INFINI_DTYPE_BF16) cudnn_dtype = CUDNN_DATA_BFLOAT16;
        else return INFINI_STATUS_BAD_TENSOR_DTYPE;
        
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(opaque->x_desc, cudnn_dtype, 4, x_dims, x_strides));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(opaque->y_desc, cudnn_dtype, 4, y_conv_dims, y_conv_strides));
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(opaque->w_desc, cudnn_dtype, CUDNN_TENSOR_NCHW, 4, w_dims));
        
        if (b) {
            int b_dims[] = {1, OC_raw, 1, 1};
            int b_strides[] = {OC_raw, 1, 1, 1};
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(opaque->b_desc, cudnn_dtype, 4, b_dims, b_strides));
        }
        
        auto pads = reinterpret_cast<const int64_t*>(pads_);
        auto strides = reinterpret_cast<const int64_t*>(strides_);
        auto dilations = reinterpret_cast<const int64_t*>(dilations_);
        
        int pad[] = {0, (int)pads[0]};
        int stride[] = {1, (int)strides[0]};
        int dilation[] = {1, (int)dilations[0]};

        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(opaque->conv_desc, 2, pad, stride, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CHECK_CUDNN(cudnnSetConvolutionGroupCount(opaque->conv_desc, groups));
        
        
        // Get workspace size
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, opaque->x_desc, opaque->w_desc, opaque->conv_desc, opaque->y_desc,
            opaque->algo, &opaque->cudnn_ws_size));

        // Total workspace: cuDNN ws + (optional) temp conv output for gated
        size_t dtype_size = (dtype == INFINI_DTYPE_F32) ? sizeof(float)
                             : (dtype == INFINI_DTYPE_F16) ? sizeof(__half)
                             : sizeof(__nv_bfloat16);
        if (info.gated()) {
            opaque->conv_out_bytes = static_cast<size_t>(B) * static_cast<size_t>(OC_raw) * static_cast<size_t>(Lout) * dtype_size;
        } else {
            opaque->conv_out_bytes = 0;
        }
        opaque->workspace_size = opaque->cudnn_ws_size + opaque->conv_out_bytes;

        *desc_ptr = new Descriptor(dtype, info, opaque->workspace_size, opaque, h->device, h->device_id);
        return INFINI_STATUS_SUCCESS;
    });
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

// CUDA kernel for gated conv1d: SiLU(A) * B where A, B are channel-wise halves
template<typename T>
__global__ void gated_conv1d_kernel(
    T* y_out,           // output [B, L, Cout]
    const T* y_conv,    // conv output [B, 2*Cout, L]
    int B, int L, int Cout
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * L * Cout;
    if (idx >= total) return;
    int b = idx / (L * Cout);
    int l = (idx % (L * Cout)) / Cout;
    int c = idx % Cout;
    
    // Access conv output: [B, 2*Cout, L] (contiguous)
    int conv_idx_a = b * (2 * Cout * L) + c * L + l;           // A channel
    int conv_idx_b = b * (2 * Cout * L) + (c + Cout) * L + l;  // B channel
    
    T a_val = y_conv[conv_idx_a];
    T b_val = y_conv[conv_idx_b];
    
    // SiLU(a) = a * sigmoid(a) = a / (1 + exp(-a))
    // Use explicit float conversion for exp function
    float a_float = static_cast<float>(a_val);
    float silu_a_float = a_float / (1.0f + expf(-a_float));
    T silu_a = static_cast<T>(silu_a_float);
    
    // Output: [B, L, Cout]
    y_out[idx] = silu_a * b_val;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w, const void *bias,
    void *stream) const
{
#ifndef ENABLE_CUDNN_API
    (void)workspace; (void)workspace_size; (void)y; (void)x; (void)w; (void)bias; (void)stream;
    return INFINI_STATUS_NOT_IMPLEMENTED;
#else
    if (workspace_size < _opaque->workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    return _opaque->internal->useCudnn(reinterpret_cast<cudaStream_t>(stream), [&](cudnnHandle_t cudnn) {
        const float alpha = 1.0f, beta = 0.0f;

        // If not gated, write conv directly into y
        if (!_info.gated()) {
            CHECK_CUDNN(cudnnConvolutionForward(
                cudnn, &alpha,
                _opaque->x_desc, x,
                _opaque->w_desc, w,
                _opaque->conv_desc, _opaque->algo,
                workspace, _opaque->cudnn_ws_size,
                &beta,
                _opaque->y_desc, y));

            if (bias && _opaque->b_desc) {
                CHECK_CUDNN(cudnnAddTensor(
                    cudnn, &alpha,
                    _opaque->b_desc, bias,
                    &alpha,
                    _opaque->y_desc, y));
            }
            return INFINI_STATUS_SUCCESS;
        }

        // Gated path: conv into temp buffer [B, 2*Cout, L]
        void* cudnn_ws = workspace;
        void* temp_conv_output = (void*)((char*)workspace + _opaque->cudnn_ws_size);

        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn, &alpha,
            _opaque->x_desc, x,
            _opaque->w_desc, w,
            _opaque->conv_desc, _opaque->algo,
            cudnn_ws, _opaque->cudnn_ws_size,
            &beta,
            _opaque->y_desc, temp_conv_output));

        if (bias && _opaque->b_desc) {
            CHECK_CUDNN(cudnnAddTensor(
                cudnn, &alpha,
                _opaque->b_desc, bias,
                &alpha,
                _opaque->y_desc, temp_conv_output));
        }

        // Launch gated activation kernel
        int conv_dims[4];
        int conv_strides[4];
        cudnnDataType_t conv_dtype;
        int ndim;
        CHECK_CUDNN(cudnnGetTensorNdDescriptor(_opaque->y_desc, 4, &conv_dtype, &ndim, conv_dims, conv_strides));
        int Bn = conv_dims[0];
        int Cout2 = conv_dims[1];
        int Ln = conv_dims[3];
        int Cout = Cout2 / 2;

        dim3 block(256);
        dim3 grid((Bn * Ln * Cout + block.x - 1) / block.x);
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

        if (conv_dtype == CUDNN_DATA_FLOAT) {
            gated_conv1d_kernel<float><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<float*>(y),
                reinterpret_cast<const float*>(temp_conv_output),
                Bn, Ln, Cout);
        } else if (conv_dtype == CUDNN_DATA_HALF) {
            gated_conv1d_kernel<__half><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<__half*>(y),
                reinterpret_cast<const __half*>(temp_conv_output),
                Bn, Ln, Cout);
        } else if (conv_dtype == CUDNN_DATA_BFLOAT16) {
            gated_conv1d_kernel<__nv_bfloat16><<<grid, block, 0, cuda_stream>>>(
                reinterpret_cast<__nv_bfloat16*>(y),
                reinterpret_cast<const __nv_bfloat16*>(temp_conv_output),
                Bn, Ln, Cout);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        return INFINI_STATUS_SUCCESS;
    });
#endif
}

} // namespace op::conv1d::nvidia
