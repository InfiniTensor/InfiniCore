#include "averagepool_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../info.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace op::averagepool_backward::cpu {

struct Descriptor::Opaque {
    device::cpu::Handle *handle;
    AvgPoolBackwardInfo info;
    size_t workspace_size = 0;

private:
    Opaque(device::cpu::Handle *handle_ptr, const AvgPoolBackwardInfo &avgpool_info)
        : handle(handle_ptr), info(avgpool_info) {
        workspace_size = 0;
    }
    
    template <typename T_out, typename T_in>
    void _avgpool_backward_1d(T_out *grad_input, const T_in *grad_output) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_width = info.input_dims[0];
        size_t output_width = info.output_dims[0];
        size_t kernel_width = info.kernel_sizes[0];
        size_t stride_width = info.strides[0];
        size_t pad_width = info.pads[0];

        const size_t input_nc_stride = input_width;
        const size_t output_nc_stride = output_width;

        size_t grad_input_nelem = info.batch * info.channels * input_width;
        memset(grad_input, 0, grad_input_nelem * sizeof(T_out));

#pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                const size_t grad_output_offset = (b * channels + c) * output_nc_stride;
                const size_t grad_input_offset = (b * channels + c) * input_nc_stride;

                for (size_t ow = 0; ow < output_width; ++ow) {
                    float grad_value = utils::cast<float>(grad_output[grad_output_offset + ow]);

                    int valid_count = 0;
                    const int window_start = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);
                    const int window_end = window_start + static_cast<int>(kernel_width);

                    for (int iw = window_start; iw < window_end; ++iw) {
                        if (iw >= 0 && iw < static_cast<int>(input_width)) {
                            valid_count++;
                        } else if (iw >= -static_cast<int>(pad_width) && 
                                   iw < static_cast<int>(input_width + pad_width)) {
                            valid_count++;
                        }
                    }
                    
                    if (valid_count > 0) {
                        float grad_distribute = grad_value / static_cast<float>(valid_count);
                        for (int iw = window_start; iw < window_end; ++iw) {
                            if (iw >= 0 && iw < static_cast<int>(input_width)) {
                                grad_input[grad_input_offset + iw] += utils::cast<T_out>(grad_distribute);
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T_out, typename T_in>
    void _avgpool_backward_2d(T_out *grad_input, const T_in *grad_output) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_height = info.input_dims[0];
        size_t input_width = info.input_dims[1];
        size_t output_height = info.output_dims[0];
        size_t output_width = info.output_dims[1];
        size_t kernel_height = info.kernel_sizes[0];
        size_t kernel_width = info.kernel_sizes[1];
        size_t stride_height = info.strides[0];
        size_t stride_width = info.strides[1];
        size_t pad_h = info.pads[0];
        size_t pad_w = info.pads[1];
        
        const size_t input_nc_stride = input_height * input_width;
        const size_t output_nc_stride = output_height * output_width;

        size_t grad_input_nelem = info.batch * info.channels * input_height * input_width;
        memset(grad_input, 0, grad_input_nelem * sizeof(T_out));

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                const size_t grad_output_offset = (b * channels + c) * output_nc_stride;
                const size_t grad_input_offset = (b * channels + c) * input_nc_stride;

                for (size_t oh = 0; oh < output_height; ++oh) {
                    for (size_t ow = 0; ow < output_width; ++ow) {
                        float grad_value = utils::cast<float>(grad_output[grad_output_offset + oh * output_width + ow]);

                        int valid_count = 0;
                        const int start_h = static_cast<int>(oh * stride_height) - static_cast<int>(pad_h);
                        const int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_w);

                        for (int kh = 0; kh < static_cast<int>(kernel_height); ++kh) {
                            for (int kw = 0; kw < static_cast<int>(kernel_width); ++kw) {
                                const int ih = start_h + kh;
                                const int iw = start_w + kw;

                                if (ih >= 0 && ih < static_cast<int>(input_height) &&
                                    iw >= 0 && iw < static_cast<int>(input_width)) {
                                    valid_count++;
                                } else if (ih >= -static_cast<int>(pad_h) && 
                                           ih < static_cast<int>(input_height + pad_h) &&
                                           iw >= -static_cast<int>(pad_w) &&
                                           iw < static_cast<int>(input_width + pad_w)) {
                                    valid_count++;
                                }
                            }
                        }

                        if (valid_count > 0) {
                            float grad_distribute = grad_value / static_cast<float>(valid_count);
                            for (int kh = 0; kh < static_cast<int>(kernel_height); ++kh) {
                                for (int kw = 0; kw < static_cast<int>(kernel_width); ++kw) {
                                    const int ih = start_h + kh;
                                    const int iw = start_w + kw;
                                    if (ih >= 0 && ih < static_cast<int>(input_height) &&
                                        iw >= 0 && iw < static_cast<int>(input_width)) {
                                        grad_input[grad_input_offset + ih * input_width + iw] += utils::cast<T_out>(grad_distribute);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T_out, typename T_in>
    void _avgpool_backward_3d(T_out *grad_input, const T_in *grad_output) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_depth = info.input_dims[0];
        size_t input_height = info.input_dims[1];
        size_t input_width = info.input_dims[2];
        size_t output_depth = info.output_dims[0];
        size_t output_height = info.output_dims[1];
        size_t output_width = info.output_dims[2];
        size_t kernel_d = info.kernel_sizes[0];
        size_t kernel_h = info.kernel_sizes[1];
        size_t kernel_w = info.kernel_sizes[2];
        size_t stride_d = info.strides[0];
        size_t stride_h = info.strides[1];
        size_t stride_w = info.strides[2];
        size_t pad_d = info.pads[0];
        size_t pad_h = info.pads[1];
        size_t pad_w = info.pads[2];

        const size_t input_nc_stride = input_depth * input_height * input_width;
        const size_t output_nc_stride = output_depth * output_height * output_width;

        size_t grad_input_nelem = info.batch * info.channels * input_depth * input_height * input_width;
        memset(grad_input, 0, grad_input_nelem * sizeof(T_out));

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                const size_t grad_output_offset = (b * channels + c) * output_nc_stride;
                const size_t grad_input_offset = (b * channels + c) * input_nc_stride;

                for (size_t od = 0; od < output_depth; ++od) {
                    for (size_t oh = 0; oh < output_height; ++oh) {
                        for (size_t ow = 0; ow < output_width; ++ow) {
                            float grad_value = utils::cast<float>(grad_output[grad_output_offset + od * output_height * output_width + oh * output_width + ow]);
                            
                            int valid_count = 0;
                            const int start_d = static_cast<int>(od * stride_d) - static_cast<int>(pad_d);
                            const int start_h = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                            const int start_w = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

                            for (int kd = 0; kd < static_cast<int>(kernel_d); ++kd) {
                                for (int kh = 0; kh < static_cast<int>(kernel_h); ++kh) {
                                    for (int kw = 0; kw < static_cast<int>(kernel_w); ++kw) {
                                        const int id = start_d + kd;
                                        const int ih = start_h + kh;
                                        const int iw = start_w + kw;

                                        if (id >= 0 && id < static_cast<int>(input_depth) &&
                                            ih >= 0 && ih < static_cast<int>(input_height) &&
                                            iw >= 0 && iw < static_cast<int>(input_width)) {
                                            valid_count++;
                                        } else if (id >= -static_cast<int>(pad_d) && 
                                                   id < static_cast<int>(input_depth + pad_d) &&
                                                   ih >= -static_cast<int>(pad_h) && 
                                                   ih < static_cast<int>(input_height + pad_h) &&
                                                   iw >= -static_cast<int>(pad_w) && 
                                                   iw < static_cast<int>(input_width + pad_w)) {
                                            valid_count++;
                                        }
                                    }
                                }
                            }
                            
                            if (valid_count > 0) {
                                float grad_distribute = grad_value / static_cast<float>(valid_count);
                                for (int kd = 0; kd < static_cast<int>(kernel_d); ++kd) {
                                    for (int kh = 0; kh < static_cast<int>(kernel_h); ++kh) {
                                        for (int kw = 0; kw < static_cast<int>(kernel_w); ++kw) {
                                            const int id = start_d + kd;
                                            const int ih = start_h + kh;
                                            const int iw = start_w + kw;
                                            if (id >= 0 && id < static_cast<int>(input_depth) &&
                                                ih >= 0 && ih < static_cast<int>(input_height) &&
                                                iw >= 0 && iw < static_cast<int>(input_width)) {
                                                grad_input[grad_input_offset + id * input_height * input_width + ih * input_width + iw] += utils::cast<T_out>(grad_distribute);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T_out, typename T_in>
    void _avgpool_backward_cpu(T_out *grad_input, const T_in *grad_output) const {
        switch (info.ndim) {
        case 1:
            _avgpool_backward_1d<T_out, T_in>(grad_input, grad_output);
            break;
        case 2:
            _avgpool_backward_2d<T_out, T_in>(grad_input, grad_output);
            break;
        case 3:
            _avgpool_backward_3d<T_out, T_in>(grad_input, grad_output);
            break;
        default:
            break;
        }
    }
public:
    Opaque(Opaque &&other) noexcept
        : handle(other.handle),
          info(std::move(other.info)),
          workspace_size(other.workspace_size) {
        other.handle = nullptr;
        other.workspace_size = 0;
    }
    
    ~Opaque() = default;

    static inline utils::Result<Opaque>
    create(device::cpu::Handle *handle_ptr,
           AvgPoolBackwardInfo &info) {
        Opaque opaque(handle_ptr, info);
        return utils::Result<Opaque>(std::move(opaque));
    }

    infiniStatus_t calculate(void *workspace, size_t workspace_size,
                             void *grad_input, const void *grad_output,
                             const void *input, infiniDtype_t dtype) const {
        if (!grad_input || !grad_output) {
            return INFINI_STATUS_BAD_PARAM;
        }
        
        size_t grad_input_nelem = info.batch * info.channels * info.input_dims[0];
        if (info.ndim > 1) {
            grad_input_nelem *= info.input_dims[1];
        }
        if (info.ndim > 2) {
            grad_input_nelem *= info.input_dims[2];
        }

        switch (dtype) {
        case INFINI_DTYPE_F32: {
            float *typed_grad_input = static_cast<float *>(grad_input);
            const float *typed_grad_output = static_cast<const float *>(grad_output);
            _avgpool_backward_cpu<float, float>(typed_grad_input, typed_grad_output);
            break;
        }
        case INFINI_DTYPE_F16: {
            float *typed_grad_input_f32 = static_cast<float*>(workspace);
            const fp16_t *typed_grad_output = static_cast<const fp16_t *>(grad_output);
            
            _avgpool_backward_cpu<float, fp16_t>(typed_grad_input_f32, typed_grad_output);
            
            fp16_t *typed_grad_input = static_cast<fp16_t*>(grad_input);
            #pragma omp parallel for
            for(size_t i = 0; i < grad_input_nelem; ++i) {
                typed_grad_input[i] = utils::cast<fp16_t>(typed_grad_input_f32[i]);
            }
            break;
        }
        case INFINI_DTYPE_BF16: {
            float *typed_grad_input_f32 = static_cast<float*>(workspace);
            const bf16_t *typed_grad_output = static_cast<const bf16_t *>(grad_output);

            _avgpool_backward_cpu<float, bf16_t>(typed_grad_input_f32, typed_grad_output);

            bf16_t *typed_grad_input = static_cast<bf16_t*>(grad_input);
            #pragma omp parallel for
            for(size_t i = 0; i < grad_input_nelem; ++i) {
                typed_grad_input[i] = utils::cast<bf16_t>(typed_grad_input_f32[i]);
            }
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        return INFINI_STATUS_SUCCESS;
    }
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

inline size_t calculateOutputSize(const AvgPoolBackwardInfo &info) {
    size_t size = info.batch * info.channels;
    for (size_t i = 0; i < info.ndim; ++i) {
        size *= info.input_dims[i];
    }
    return size;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t input_desc,
    void *kernel_size,
    void *strides,
    void *pads,
    bool ceil_mode) {
    
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = grad_input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    auto result = AvgPoolBackwardInfo::create(
        grad_input_desc, grad_output_desc, input_desc, kernel_size, strides, pads, ceil_mode);
    CHECK_RESULT(result);
    auto info = result.take();

    auto opaque_result = Opaque::create(handle, info);
    CHECK_RESULT(opaque_result);
    auto opaque = new Opaque(opaque_result.take());

    size_t workspace_size = 0;
    if (dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16) {
        workspace_size = calculateOutputSize(info) * sizeof(float);
    }
    
    *desc_ptr = new Descriptor(dtype, std::move(info), workspace_size,
                             opaque, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    const void *grad_output,
    const void *input,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    
    return _opaque->calculate(workspace, workspace_size, grad_input, grad_output, input, _dtype);
}

} // namespace op::averagepool_backward::cpu
