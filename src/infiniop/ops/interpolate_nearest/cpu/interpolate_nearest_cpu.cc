#include "interpolate_nearest_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../info.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace op::interpolate_nearest::cpu {

struct Descriptor::Opaque {
    device::cpu::Handle *handle;
    InterpolateNearestInfo info;
    size_t workspace_size = 0;

private:
    Opaque(device::cpu::Handle *handle_ptr, const InterpolateNearestInfo &interpolate_info)
        : handle(handle_ptr), info(interpolate_info) {
        workspace_size = 0;
    }

    template <typename T>
    size_t compute_input_index_1d(size_t idx) const {
        size_t temp = idx;

        // 1D插值：3D张量 (N, C, W)
        size_t w = temp % info.output_size[0];
        temp /= info.output_size[0];
        size_t c = temp % info.channels;
        size_t b = temp / info.channels;

        float inv_scale = static_cast<float>(info.input_size[0]) / static_cast<float>(info.output_size[0]);
        size_t input_w = std::min(static_cast<size_t>(std::floor(static_cast<float>(w) * inv_scale)),
                                  info.input_size[0] - 1);

        return b * info.input_stride[0] + c * info.input_stride[1] + input_w * info.input_stride[2];
    }

    // 计算2D插值的输入索引
    template <typename T>
    size_t compute_input_index_2d(size_t idx) const {
        size_t temp = idx;

        // 2D插值：4D张量 (N, C, H, W)
        size_t w = temp % info.output_size[1]; // width在索引1
        temp /= info.output_size[1];
        size_t h = temp % info.output_size[0]; // height在索引0
        temp /= info.output_size[0];
        size_t c = temp % info.channels;
        size_t b = temp / info.channels;

        float inv_scale_h = static_cast<float>(info.input_size[0]) / static_cast<float>(info.output_size[0]);
        float inv_scale_w = static_cast<float>(info.input_size[1]) / static_cast<float>(info.output_size[1]);

        size_t input_h = std::min(static_cast<size_t>(std::floor(static_cast<float>(h) * inv_scale_h)),
                                  info.input_size[0] - 1);
        size_t input_w = std::min(static_cast<size_t>(std::floor(static_cast<float>(w) * inv_scale_w)),
                                  info.input_size[1] - 1);

        return b * info.input_stride[0] + c * info.input_stride[1] + input_h * info.input_stride[2] + input_w * info.input_stride[3];
    }

    // 计算3D插值的输入索引
    template <typename T>
    size_t compute_input_index_3d(size_t idx) const {
        size_t temp = idx;

        // 3D插值：5D张量 (N, C, D, H, W)
        size_t w = temp % info.output_size[2]; // width在索引2
        temp /= info.output_size[2];
        size_t h = temp % info.output_size[1]; // height在索引1
        temp /= info.output_size[1];
        size_t d = temp % info.output_size[0]; // depth在索引0
        temp /= info.output_size[0];
        size_t c = temp % info.channels;
        size_t b = temp / info.channels;

        float inv_scale_d = static_cast<float>(info.input_size[0]) / static_cast<float>(info.output_size[0]);
        float inv_scale_h = static_cast<float>(info.input_size[1]) / static_cast<float>(info.output_size[1]);
        float inv_scale_w = static_cast<float>(info.input_size[2]) / static_cast<float>(info.output_size[2]);

        size_t input_d = std::min(static_cast<size_t>(std::floor(static_cast<float>(d) * inv_scale_d)),
                                  info.input_size[0] - 1);
        size_t input_h = std::min(static_cast<size_t>(std::floor(static_cast<float>(h) * inv_scale_h)),
                                  info.input_size[1] - 1);
        size_t input_w = std::min(static_cast<size_t>(std::floor(static_cast<float>(w) * inv_scale_w)),
                                  info.input_size[2] - 1);

        return b * info.input_stride[0] + c * info.input_stride[1] + input_d * info.input_stride[2] + input_h * info.input_stride[3] + input_w * info.input_stride[4];
    }

    // 计算输出索引
    template <typename T>
    size_t compute_output_index(size_t idx) const {
        size_t temp = idx;
        size_t w, h, d, c, b;

        switch (info.dim) {
        case INTERPOLATE_1D: {
            // 3D张量 (N, C, W)
            w = temp % info.output_size[0];
            temp /= info.output_size[0];
            c = temp % info.channels;
            b = temp / info.channels;
            return b * info.output_stride[0] + c * info.output_stride[1] + w * info.output_stride[2];
        }

        case INTERPOLATE_2D: {
            // 4D张量 (N, C, H, W)
            w = temp % info.output_size[1];
            temp /= info.output_size[1];
            h = temp % info.output_size[0];
            temp /= info.output_size[0];
            c = temp % info.channels;
            b = temp / info.channels;
            return b * info.output_stride[0] + c * info.output_stride[1] + h * info.output_stride[2] + w * info.output_stride[3];
        }

        case INTERPOLATE_3D: {
            // 5D张量 (N, C, D, H, W)
            w = temp % info.output_size[2];
            temp /= info.output_size[2];
            h = temp % info.output_size[1];
            temp /= info.output_size[1];
            d = temp % info.output_size[0];
            temp /= info.output_size[0];
            c = temp % info.channels;
            b = temp / info.channels;
            return b * info.output_stride[0] + c * info.output_stride[1] + d * info.output_stride[2] + h * info.output_stride[3] + w * info.output_stride[4];
        }

        default:
            return 0;
        }
    }

    // 计算总元素数
    size_t calculate_total_elements() const {
        size_t total = info.batch_size * info.channels;
        switch (info.dim) {
        case INTERPOLATE_1D:
            total *= info.output_size[0]; // width
            break;
        case INTERPOLATE_2D:
            total *= info.output_size[0] * info.output_size[1]; // height * width
            break;
        case INTERPOLATE_3D:
            total *= info.output_size[0] * info.output_size[1] * info.output_size[2]; // depth * height * width
            break;
        }
        return total;
    }

    // 主要的插值计算函数
    template <typename T>
    void interpolate_nearest_cpu(T *output, const T *input) const {
        size_t total_elements = calculate_total_elements();

#pragma omp parallel for schedule(static)
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(total_elements); ++idx) {
            size_t input_idx;

            switch (info.dim) {
            case INTERPOLATE_1D:
                input_idx = compute_input_index_1d<T>(idx);
                break;
            case INTERPOLATE_2D:
                input_idx = compute_input_index_2d<T>(idx);
                break;
            case INTERPOLATE_3D:
                input_idx = compute_input_index_3d<T>(idx);
                break;
            default:
                continue;
            }

            size_t output_idx = compute_output_index<T>(idx);
            output[output_idx] = input[input_idx];
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
           const InterpolateNearestInfo &info,
           infiniDtype_t data_type) {
        if (data_type != INFINI_DTYPE_F32 && data_type != INFINI_DTYPE_F16 && data_type != INFINI_DTYPE_BF16 && data_type != INFINI_DTYPE_I8) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        Opaque opaque(handle_ptr, info);
        return utils::Result<Opaque>(std::move(opaque));
    }

    infiniStatus_t calculate(void *workspace, size_t workspace_size,
                             void *output, const void *input, infiniDtype_t dtype) const {

        if (!output || !input) {
            return INFINI_STATUS_BAD_PARAM;
        }

        switch (dtype) {
        case INFINI_DTYPE_F32: {
            float *typed_output = static_cast<float *>(output);
            const float *typed_input = static_cast<const float *>(input);
            interpolate_nearest_cpu(typed_output, typed_input);
            break;
        }

        case INFINI_DTYPE_F16: {
            fp16_t *typed_output = static_cast<fp16_t *>(output);
            const fp16_t *typed_input = static_cast<const fp16_t *>(input);
            interpolate_nearest_cpu(typed_output, typed_input);
            break;
        }

        case INFINI_DTYPE_BF16: {
            bf16_t *typed_output = static_cast<bf16_t *>(output);
            const bf16_t *typed_input = static_cast<const bf16_t *>(input);
            interpolate_nearest_cpu(typed_output, typed_input);
            break;
        }

        case INFINI_DTYPE_I8: {
            int8_t *typed_output = static_cast<int8_t *>(output);
            const int8_t *typed_input = static_cast<const int8_t *>(input);
            interpolate_nearest_cpu(typed_output, typed_input);
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

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t output_desc,
                                  infiniopTensorDescriptor_t input_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = output_desc->dtype();

    // 检查数据类型支持
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_I8);

    InterpolateNearestInfo info;
    CHECK_STATUS(InterpolateNearestInfo::create(&info, output_desc, input_desc));

    auto opaque_result = Opaque::create(handle, info, dtype);
    CHECK_RESULT(opaque_result);
    auto opaque = new Opaque(opaque_result.take());

    *desc_ptr = new Descriptor(dtype, info, opaque->workspace_size, opaque,
                               handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *output, const void *input,
                                     void *stream) const {
    return _opaque->calculate(workspace, workspace_size, output, input, _dtype);
}

} // namespace op::interpolate_nearest::cpu
