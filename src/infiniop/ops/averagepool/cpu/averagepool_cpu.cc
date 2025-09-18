#include "averagepool_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../info.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace op::averagepool::cpu {

struct Descriptor::Opaque {
    device::cpu::Handle *handle;
    AvgPoolInfo info;
    size_t workspace_size = 0;

private:
    Opaque(device::cpu::Handle *handle_ptr, const AvgPoolInfo &avgpool_info)
        : handle(handle_ptr), info(avgpool_info) {
        workspace_size = 0;
    }

    template <typename T, typename Ydata>
    void _avgpool_1d(Ydata *output, const T *input) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_width = info.input_dims[0];
        size_t output_width = info.output_dims[0];
        size_t kernel_width = info.kernel_sizes[0];
        size_t stride_width = info.strides[0];
        size_t pad_width = info.pads[0];

        const size_t input_nc_stride = input_width;
        const size_t output_nc_stride = output_width;

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                const size_t input_offset = (b * channels + c) * input_nc_stride;
                const size_t output_offset = (b * channels + c) * output_nc_stride;

                for (size_t ow = 0; ow < output_width; ++ow) {
                    float sum = 0.0f;
                    int valid_count = 0;

                    const int window_start = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);
                    const int window_end = window_start + static_cast<int>(kernel_width);

                    for (int iw = window_start; iw < window_end; ++iw) {
                        if (iw >= 0 && iw < static_cast<int>(input_width)) {
                            sum += utils::cast<float>(input[input_offset + iw]);
                            valid_count++;
                        } else if (iw >= -static_cast<int>(pad_width) && 
                                   iw < static_cast<int>(input_width + pad_width)) {
                            valid_count++;
                        }
                    }

                    float result = 0.0f;
                    if (valid_count > 0) {
                        result = sum / static_cast<float>(valid_count);
                    }
                    output[output_offset + ow] = utils::cast<Ydata>(result);
                }
            }
        }
    }    
    
    template <typename T, typename Ydata>
    void _avgpool_2d(Ydata *output, const T *input) const {
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
        size_t pad_height = info.pads[0];
        size_t pad_width = info.pads[1];

        const size_t input_nc_stride = input_height * input_width;
        const size_t output_nc_stride = output_height * output_width;

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                const size_t input_offset = (b * channels + c) * input_nc_stride;
                const size_t output_offset = (b * channels + c) * output_nc_stride;

                for (size_t oh = 0; oh < output_height; ++oh) {
                    for (size_t ow = 0; ow < output_width; ++ow) {
                        float sum = 0.0f;
                        int valid_count = 0;

                        const int start_h = static_cast<int>(oh * stride_height) - static_cast<int>(pad_height);
                        const int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);

                        for (int kh = 0; kh < static_cast<int>(kernel_height); ++kh) {
                            for (int kw = 0; kw < static_cast<int>(kernel_width); ++kw) {
                                const int ih = start_h + kh;
                                const int iw = start_w + kw;

                                if (ih >= 0 && ih < static_cast<int>(input_height) &&
                                    iw >= 0 && iw < static_cast<int>(input_width)) {
                                    sum += utils::cast<float>(input[input_offset + ih * input_width + iw]);
                                    valid_count++;
                                } else if (ih >= -static_cast<int>(pad_height) && 
                                           ih < static_cast<int>(input_height + pad_height) &&
                                           iw >= -static_cast<int>(pad_width) &&
                                           iw < static_cast<int>(input_width + pad_width)) {
                                    valid_count++;
                                }
                            }
                        }

                        float result = 0.0f;
                        if (valid_count > 0) {
                            result = sum / static_cast<float>(valid_count);
                        }
                        output[output_offset + oh * output_width + ow] = utils::cast<Ydata>(result);
                    }
                }
            }
        }
    }

    template <typename T, typename Ydata>
    void _avgpool_3d(Ydata *output, const T *input) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_depth = info.input_dims[0];
        size_t input_height = info.input_dims[1];
        size_t input_width = info.input_dims[2];
        size_t output_depth = info.output_dims[0];
        size_t output_height = info.output_dims[1];
        size_t output_width = info.output_dims[2];
        size_t kernel_depth = info.kernel_sizes[0];
        size_t kernel_height = info.kernel_sizes[1];
        size_t kernel_width = info.kernel_sizes[2];
        size_t stride_depth = info.strides[0];
        size_t stride_height = info.strides[1];
        size_t stride_width = info.strides[2];
        size_t pad_depth = info.pads[0];
        size_t pad_height = info.pads[1];
        size_t pad_width = info.pads[2];

        const size_t input_nc_stride = input_depth * input_height * input_width;
        const size_t output_nc_stride = output_depth * output_height * output_width;

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                const size_t input_offset = (b * channels + c) * input_nc_stride;
                const size_t output_offset = (b * channels + c) * output_nc_stride;

                for (size_t od = 0; od < output_depth; ++od) {
                    for (size_t oh = 0; oh < output_height; ++oh) {
                        for (size_t ow = 0; ow < output_width; ++ow) {
                            float sum = 0.0f;
                            int valid_count = 0;

                            const int start_d = static_cast<int>(od * stride_depth) - static_cast<int>(pad_depth);
                            const int start_h = static_cast<int>(oh * stride_height) - static_cast<int>(pad_height);
                            const int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);

                            for (int kd = 0; kd < static_cast<int>(kernel_depth); ++kd) {
                                const int id = start_d + kd;
                                for (int kh = 0; kh < static_cast<int>(kernel_height); ++kh) {
                                    const int ih = start_h + kh;
                                    for (int kw = 0; kw < static_cast<int>(kernel_width); ++kw) {
                                        const int iw = start_w + kw;

                                        if (id >= 0 && id < static_cast<int>(input_depth) &&
                                            ih >= 0 && ih < static_cast<int>(input_height) &&
                                            iw >= 0 && iw < static_cast<int>(input_width)) {
                                            const size_t idx = id * (input_height * input_width) + 
                                                            ih * input_width + iw;
                                            sum += utils::cast<float>(input[input_offset + idx]);
                                            valid_count++;
                                        } else if (id >= -static_cast<int>(pad_depth) && 
                                                   id < static_cast<int>(input_depth + pad_depth) &&
                                                   ih >= -static_cast<int>(pad_height) && 
                                                   ih < static_cast<int>(input_height + pad_height) &&
                                                   iw >= -static_cast<int>(pad_width) && 
                                                   iw < static_cast<int>(input_width + pad_width)) {
                                            valid_count++;
                                        }
                                    }
                                }
                            }

                            float result = 0.0f;
                            if (valid_count > 0) {
                                result = sum / static_cast<float>(valid_count);
                            }
                            
                            const size_t out_idx = od * (output_height * output_width) + 
                                                oh * output_width + ow;
                            output[output_offset + out_idx] = utils::cast<Ydata>(result);
                        }
                    }
                }
            }
        }
    }

    template <typename T, typename Ydata>
    void _avgpool_cpu(Ydata *output, const T *input) const {
        switch (info.ndim) {
        case 1:
            _avgpool_1d<T, Ydata>(output, input);
            break;
        case 2:
            _avgpool_2d<T, Ydata>(output, input);
            break;
        case 3:
            _avgpool_3d<T, Ydata>(output, input);
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
           AvgPoolInfo &info) {

        Opaque opaque(handle_ptr, info);
        return utils::Result<Opaque>(std::move(opaque));
    }

    infiniStatus_t calculate(void *workspace, size_t workspace_size,
                            void *output, const void *input, infiniDtype_t dtype) const {
        if (!output || !input) {
            return INFINI_STATUS_BAD_PARAM;
        }
        
        size_t output_size = info.batch * info.channels;
        for (size_t i = 0; i < info.ndim; ++i) {
            output_size *= info.output_dims[i];
        }

        switch (dtype) {
        case INFINI_DTYPE_F32: {
            float *typed_output = static_cast<float *>(output);
            const float *typed_input = static_cast<const float *>(input);
            _avgpool_cpu<float, float>(typed_output, typed_input);
            break;
        }
        case INFINI_DTYPE_F16: {
            float *typed_output_f32 = static_cast<float*>(workspace);
            const fp16_t *typed_input = static_cast<const fp16_t *>(input);
            
            _avgpool_cpu<fp16_t, float>(typed_output_f32, typed_input);
            
            fp16_t *typed_output = static_cast<fp16_t*>(output);
            #pragma omp parallel for
            for(size_t i = 0; i < output_size; ++i) {
                typed_output[i] = utils::cast<fp16_t>(typed_output_f32[i]);
            }
            break;
        }
        case INFINI_DTYPE_BF16: {
            float *typed_output_f32 = static_cast<float*>(workspace);
            const bf16_t *typed_input = static_cast<const bf16_t *>(input);

            _avgpool_cpu<bf16_t, float>(typed_output_f32, typed_input);

            bf16_t *typed_output = static_cast<bf16_t*>(output);
            #pragma omp parallel for
            for(size_t i = 0; i < output_size; ++i) {
                typed_output[i] = utils::cast<bf16_t>(typed_output_f32[i]);
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

inline size_t calculateOutputSize(const AvgPoolInfo &info) {
    size_t size = info.batch * info.channels;
    for(size_t i = 0; i < info.ndim; ++i) {
        size *= info.output_dims[i];
    }
    return size;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    void *kernel_size,
    void *strides,
    void *pads,
    bool ceil_mode) {
    
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    auto result = AvgPoolInfo::create(output_desc, input_desc, kernel_size,
                                    strides, pads, ceil_mode);
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
    void *output,
    const void *input,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    
    return _opaque->calculate(workspace, workspace_size, output, input, _dtype);
}

} // namespace op::averagepool::cpu
