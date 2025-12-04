#include "aminmax_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../tensor.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace op::aminmax::cpu {

// 辅助函数：将线性索引转换为多维索引（与 any 相同）
namespace {
inline void linear_to_multi_index(
    size_t linear_idx,
    const std::vector<size_t> &shape,
    std::vector<size_t> &indices) {
    size_t temp = linear_idx;
    for (size_t i = shape.size(); i-- > 0;) {
        indices[i] = temp % shape[i];
        temp /= shape[i];
    }
}

// 笛卡尔积迭代器（与 any 相同）
class CartesianProductIterator {
public:
    CartesianProductIterator(const std::vector<size_t> &dims)
        : dims(dims), indices(dims.size(), 0), first_call(true) {}

    bool next() {
        if (first_call) {
            first_call = false;
            return !dims.empty();
        }

        for (int i = dims.size() - 1; i >= 0; --i) {
            indices[i]++;
            if (indices[i] < dims[i]) {
                return true;
            }
            indices[i] = 0;
        }
        return false;
    }

    const std::vector<size_t> &get() const {
        return indices;
    }

private:
    std::vector<size_t> dims;
    std::vector<size_t> indices;
    bool first_call;
};
} // namespace

// 全局 aminmax kernel
template <typename T>
void aminmax_global_kernel(
    const T *input_data,
    T *min_output_data,
    T *max_output_data,
    const AminmaxInfo &info) {

    // 对于 fp16_t 和 bf16_t，使用 float 来计算以提高精度
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        float min_val_f = std::numeric_limits<float>::max();
        float max_val_f = std::numeric_limits<float>::lowest();

        size_t in_numel = 1;
        for (auto s : info.in_shape) {
            in_numel *= s;
        }

        for (size_t i = 0; i < in_numel; ++i) {
            std::vector<size_t> in_indices(info.in_shape.size());
            linear_to_multi_index(i, info.in_shape, in_indices);

            size_t in_offset = 0;
            for (size_t j = 0; j < in_indices.size(); ++j) {
                in_offset += in_indices[j] * info.in_strides[j];
            }

            // 转换为 float 进行比较
            float val_f = utils::cast<float>(input_data[in_offset]);
            if (val_f < min_val_f) min_val_f = val_f;
            if (val_f > max_val_f) max_val_f = val_f;
        }

        min_output_data[0] = utils::cast<T>(min_val_f);
        max_output_data[0] = utils::cast<T>(max_val_f);
    } else {
        T min_val = std::numeric_limits<T>::max();
        T max_val = std::numeric_limits<T>::lowest();

        size_t in_numel = 1;
        for (auto s : info.in_shape) {
            in_numel *= s;
        }

        for (size_t i = 0; i < in_numel; ++i) {
            std::vector<size_t> in_indices(info.in_shape.size());
            linear_to_multi_index(i, info.in_shape, in_indices);

            size_t in_offset = 0;
            for (size_t j = 0; j < in_indices.size(); ++j) {
                in_offset += in_indices[j] * info.in_strides[j];
            }

            T val = input_data[in_offset];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        min_output_data[0] = min_val;
        max_output_data[0] = max_val;
    }
}

// 通用 aminmax kernel（支持多维规约、非连续张量）
template <typename T>
void aminmax_kernel(
    const T *input_data,
    T *min_output_data,
    T *max_output_data,
    const AminmaxInfo &info) {

    // 处理全局 aminmax
    if (info.is_global) {
        aminmax_global_kernel(input_data, min_output_data, max_output_data, info);
        return;
    }

    size_t out_numel = 1;
    for (auto s : info.out_shape) {
        out_numel *= s;
    }

    for (size_t out_idx = 0; out_idx < out_numel; ++out_idx) {
        // 1. 将 flat 输出索引转换为多维索引
        std::vector<size_t> out_indices(info.out_shape.size());
        linear_to_multi_index(out_idx, info.out_shape, out_indices);

        // 2. 构建完整的输入索引（根据 keepdim 和 reduce_dims）
        std::vector<size_t> in_indices(info.in_shape.size());
        if (info.keepdim) {
            in_indices = out_indices;
        } else {
            size_t out_index = 0;
            for (size_t i = 0; i < info.in_shape.size(); ++i) {
                bool is_reduce_dim = false;
                for (auto rd : info.reduce_dims) {
                    if (i == static_cast<size_t>(rd)) {
                        is_reduce_dim = true;
                        break;
                    }
                }
                if (!is_reduce_dim) {
                    in_indices[i] = out_indices[out_index++];
                }
            }
        }

        // 3. 构建笛卡尔积：遍历所有 reduce_dims 的组合
        std::vector<size_t> reduce_dims_sizes;
        for (auto rd : info.reduce_dims) {
            reduce_dims_sizes.push_back(info.in_shape[rd]);
        }

        T min_val;
        T max_val;

        // 对于 fp16_t 和 bf16_t，使用 float 来计算以提高精度
        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            float min_val_f = std::numeric_limits<float>::max();
            float max_val_f = std::numeric_limits<float>::lowest();

            CartesianProductIterator iter(reduce_dims_sizes);
            while (iter.next()) {
                // 构建完整的输入索引
                std::vector<size_t> full_in_indices = in_indices;
                size_t reduce_index = 0;
                for (auto rd : info.reduce_dims) {
                    full_in_indices[rd] = iter.get()[reduce_index++];
                }

                // 计算输入内存偏移
                size_t in_offset = 0;
                for (size_t i = 0; i < full_in_indices.size(); ++i) {
                    in_offset += full_in_indices[i] * info.in_strides[i];
                }

                // 转换为 float 进行比较
                float val_f = utils::cast<float>(input_data[in_offset]);
                if (val_f < min_val_f) min_val_f = val_f;
                if (val_f > max_val_f) max_val_f = val_f;
            }
            
            // 转换回 T 类型
            min_val = utils::cast<T>(min_val_f);
            max_val = utils::cast<T>(max_val_f);
        } else {
            min_val = std::numeric_limits<T>::max();
            max_val = std::numeric_limits<T>::lowest();

            CartesianProductIterator iter(reduce_dims_sizes);
            while (iter.next()) {
                // 构建完整的输入索引
                std::vector<size_t> full_in_indices = in_indices;
                size_t reduce_index = 0;
                for (auto rd : info.reduce_dims) {
                    full_in_indices[rd] = iter.get()[reduce_index++];
                }

                // 计算输入内存偏移
                size_t in_offset = 0;
                for (size_t i = 0; i < full_in_indices.size(); ++i) {
                    in_offset += full_in_indices[i] * info.in_strides[i];
                }

                // 更新 min 和 max
                T val = input_data[in_offset];
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
        
        // 4. 写入输出（考虑输出张量的 strides）
        size_t min_out_offset = 0;
        size_t max_out_offset = 0;
        for (size_t i = 0; i < out_indices.size(); ++i) {
            min_out_offset += out_indices[i] * info.out_strides[i];
            max_out_offset += out_indices[i] * info.out_strides[i];
        }

        min_output_data[min_out_offset] = min_val;
        max_output_data[max_out_offset] = max_val;
    }
}

// 创建 AminmaxInfo（与 AnyInfo::create 类似）
utils::Result<AminmaxInfo> AminmaxInfo::create(
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int64_t dim,
    int32_t keepdim,
    int32_t has_dim) {

    AminmaxInfo info;
    info.keepdim = (keepdim != 0);
    info.dtype = input_desc->dtype();

    // 获取输入形状和 strides
    size_t ndim = input_desc->ndim();
    info.in_shape.resize(ndim);
    info.in_strides.resize(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        info.in_shape[i] = input_desc->dim(i);
        info.in_strides[i] = input_desc->stride(i);
    }

    // 处理 reduce_dims
    if (has_dim) {
        int64_t d = dim;
        if (d < 0) {
            d += static_cast<int64_t>(ndim);
        }
        if (d < 0 || d >= static_cast<int64_t>(ndim)) {
            return utils::Result<AminmaxInfo>(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }
        info.reduce_dims.push_back(d);
    }

    info.is_global = info.reduce_dims.empty();

    // 获取输出形状和 strides
    size_t out_ndim = output_desc->ndim();
    info.out_shape.resize(out_ndim);
    info.out_strides.resize(out_ndim);
    for (size_t i = 0; i < out_ndim; ++i) {
        info.out_shape[i] = output_desc->dim(i);
        info.out_strides[i] = output_desc->stride(i);
    }

    return utils::Result<AminmaxInfo>(info);
}

// 实际计算函数（根据 dtype 分发）
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *min_output,
    void *max_output,
    const void *input,
    void *stream) const {

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        aminmax_kernel(static_cast<const fp16_t *>(input),
                      static_cast<fp16_t *>(min_output),
                      static_cast<fp16_t *>(max_output), _info);
        break;
    case INFINI_DTYPE_BF16:
        aminmax_kernel(static_cast<const bf16_t *>(input),
                      static_cast<bf16_t *>(min_output),
                      static_cast<bf16_t *>(max_output), _info);
        break;
    case INFINI_DTYPE_F32:
        aminmax_kernel(static_cast<const float *>(input),
                      static_cast<float *>(min_output),
                      static_cast<float *>(max_output), _info);
        break;
    case INFINI_DTYPE_F64:
        aminmax_kernel(static_cast<const double *>(input),
                      static_cast<double *>(min_output),
                      static_cast<double *>(max_output), _info);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t min_output_desc,
    infiniopTensorDescriptor_t max_output_desc,
    infiniopTensorDescriptor_t input_desc,
    int64_t dim,
    int32_t keepdim,
    int32_t has_dim) {

    // 验证 min 和 max 输出形状相同
    if (min_output_desc->shape() != max_output_desc->shape()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto result = AminmaxInfo::create(
        input_desc, min_output_desc, dim, keepdim, has_dim);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::aminmax::cpu

