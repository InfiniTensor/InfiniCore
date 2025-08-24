#include "rms_norm_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../info.h"
#include "infiniop.h"
#include <cmath>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <omp.h>

namespace op::rms_norm_backward::cpu {

// 直接从long double转换到目标类型，避免中间float损耗
template<typename T>
T directCast(long double val) {
    if constexpr (std::is_same<T, bf16_t>::value) {
        // BF16数值范围限制
        val = std::max(static_cast<long double>(-65504.0), std::min(static_cast<long double>(65504.0), val));
        return utils::cast<bf16_t>(static_cast<float>(val));
    } else if constexpr (std::is_same<T, fp16_t>::value) {
        // F16数值范围限制
        val = std::max(static_cast<long double>(-65504.0), std::min(static_cast<long double>(65504.0), val));
        return utils::cast<fp16_t>(static_cast<float>(val));
    } else {
        return static_cast<T>(val);
    }
}

// 高精度类型转换
template<typename T>
float preciseCast(const T& val) {
    if constexpr (std::is_same<T, bf16_t>::value) {
        return utils::cast<float>(val);
    } else if constexpr (std::is_same<T, fp16_t>::value) {
        return utils::cast<float>(val);
    } else {
        return static_cast<float>(val);
    }
}

// 超高精度类型转换，用于中间计算
template<typename T>
long double ultraPreciseCast(const T& val) {
    if constexpr (std::is_same<T, bf16_t>::value) {
        return static_cast<long double>(utils::cast<float>(val));
    } else if constexpr (std::is_same<T, fp16_t>::value) {
        return static_cast<long double>(utils::cast<float>(val));
    } else {
        return static_cast<long double>(val);
    }
}

// Kahan求和算法，提高数值精度
struct EnhancedKahanSum {
    long double sum = 0.0L;
    long double c = 0.0L; // 补偿项
    
    void add(long double value) {
        long double y = value - c;
        long double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    long double get() const {
        return sum;
    }
};

// 自适应epsilon计算
template<typename T>
float computeAdaptiveEpsilon(const T* data, size_t size, float base_epsilon = 1e-6f) {
    // 计算数据的动态范围
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < size; ++i) {
        float val = preciseCast(data[i]);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    float range = max_val - min_val;
    // 根据数据范围调整epsilon
    return std::max(base_epsilon, range * 1e-8f);
}

// 稳定的RMS计算
template<typename T>
float computeStableRMS(const T* data, size_t size, float epsilon) {
    EnhancedKahanSum sum_squares;
    
    for (size_t i = 0; i < size; ++i) {
        long double val = ultraPreciseCast(data[i]);
        sum_squares.add(val * val);
    }
    
    long double mean_square = sum_squares.get() / static_cast<long double>(size);
    return static_cast<float>(std::sqrt(mean_square + static_cast<long double>(epsilon)));
}


template<typename T>
infiniStatus_t rmsNormBackwardImpl(
    void *grad_x,
    void *grad_w,
    const void *grad_y,
    const void *x,
    const void *w,
    const RMSNormBackwardInfo &info) {
    
    const T *grad_y_ptr = reinterpret_cast<const T *>(grad_y);
    const T *x_ptr = reinterpret_cast<const T *>(x);
    const T *w_ptr = reinterpret_cast<const T *>(w);
    T *grad_x_ptr = reinterpret_cast<T *>(grad_x);
    T *grad_w_ptr = reinterpret_cast<T *>(grad_w);
    
    size_t batch_size = info.batch_size();
    size_t norm_size = info.dim();
    
    // 使用传入的epsilon值
    float epsilon = info.epsilon;
    
    // 初始化grad_w为0
    for (size_t i = 0; i < norm_size; ++i) {
        grad_w_ptr[i] = T{};
    }
    
    // 使用线程局部存储优化grad_w并行累加
    std::vector<std::vector<long double>> thread_local_grad_w;
    
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        // 确保thread_local_grad_w有足够的空间
#pragma omp single
        {
            thread_local_grad_w.resize(num_threads, std::vector<long double>(norm_size, 0.0L));
        }
        
#pragma omp for
        for (size_t batch = 0; batch < batch_size; ++batch) {
            // 计算当前batch的多维索引（行优先顺序）
            std::vector<size_t> batch_indices(info.ndim() - 1);
            size_t remaining = batch;
            for (int dim = static_cast<int>(info.ndim()) - 2; dim >= 0; --dim) {
                batch_indices[dim] = remaining % info.shape[dim];
                remaining /= info.shape[dim];
            }
            
            // 计算当前batch在各个张量中的起始偏移
            size_t x_batch_offset = 0;
            size_t grad_y_batch_offset = 0;
            size_t grad_x_batch_offset = 0;
            for (size_t dim = 0; dim < info.ndim() - 1; ++dim) {
                x_batch_offset += batch_indices[dim] * info.x_strides[dim];
                grad_y_batch_offset += batch_indices[dim] * info.grad_y_strides[dim];
                grad_x_batch_offset += batch_indices[dim] * info.grad_x_strides[dim];
            }
            
            // 使用strides计算RMS
            EnhancedKahanSum sum_squares;
            for (size_t i = 0; i < norm_size; ++i) {
                size_t x_idx = x_batch_offset + i * info.x_strides[info.ndim() - 1];
                long double x_val = ultraPreciseCast(x_ptr[x_idx]);
                sum_squares.add(x_val * x_val);
            }
            long double mean_square = sum_squares.get() / static_cast<long double>(norm_size);
            float rms = static_cast<float>(std::sqrt(mean_square + static_cast<long double>(epsilon)));
            
            // 计算RMS²用于梯度计算
            long double rms_squared = static_cast<long double>(rms) * static_cast<long double>(rms);
            long double rms_cubed = rms_squared * static_cast<long double>(rms);
            
            // 使用Kahan求和计算sum(grad_y * w * x)
            EnhancedKahanSum sum_grad_y_w_x;
            for (size_t i = 0; i < norm_size; ++i) {
                size_t x_idx = x_batch_offset + i * info.x_strides[info.ndim() - 1];
                size_t grad_y_idx = grad_y_batch_offset + i * info.grad_y_strides[info.ndim() - 1];
                size_t w_idx = i * info.w_strides[0];
                
                long double gy = ultraPreciseCast(grad_y_ptr[grad_y_idx]);
                long double w_val = ultraPreciseCast(w_ptr[w_idx]);
                long double x_val = ultraPreciseCast(x_ptr[x_idx]);
                sum_grad_y_w_x.add(gy * w_val * x_val);
            }
            
            long double sum_value = sum_grad_y_w_x.get();
            
            // 计算梯度
            for (size_t i = 0; i < norm_size; ++i) {
                size_t x_idx = x_batch_offset + i * info.x_strides[info.ndim() - 1];
                size_t grad_y_idx = grad_y_batch_offset + i * info.grad_y_strides[info.ndim() - 1];
                size_t grad_x_idx = grad_x_batch_offset + i * info.grad_x_strides[info.ndim() - 1];
                size_t w_idx = i * info.w_strides[0];
                
                long double gy = ultraPreciseCast(grad_y_ptr[grad_y_idx]);
                long double w_val = ultraPreciseCast(w_ptr[w_idx]);
                long double x_val = ultraPreciseCast(x_ptr[x_idx]);
                
                // RMSNorm grad_x计算: (w * grad_y) / rms - (x * sum(grad_y * w * x)) / (norm_size * rms³)
                long double gx = (w_val * gy) / static_cast<long double>(rms) - 
                               (x_val * sum_value) / (static_cast<long double>(norm_size) * rms_cubed);
                grad_x_ptr[grad_x_idx] = directCast<T>(gx);
                
                // grad_w累加到线程局部存储: grad_w = (x * grad_y) / rms
                long double gw = (x_val * gy) / static_cast<long double>(rms);
                thread_local_grad_w[thread_id][i] += gw;
            }
        }
    }
    
    // 合并所有线程的grad_w结果
    for (size_t i = 0; i < norm_size; ++i) {
        long double total_gw = 0.0L;
        for (const auto& local_grad_w : thread_local_grad_w) {
            total_gw += local_grad_w[i];
        }
        grad_w_ptr[i] = directCast<T>(total_gw);
    }
    
    return INFINI_STATUS_SUCCESS;
}

// 显式模板实例化
template infiniStatus_t rmsNormBackwardImpl<float>(
    void *, void *, const void *, const void *, const void *, const RMSNormBackwardInfo &);
template infiniStatus_t rmsNormBackwardImpl<fp16_t>(
    void *, void *, const void *, const void *, const void *, const RMSNormBackwardInfo &);
template infiniStatus_t rmsNormBackwardImpl<bf16_t>(
    void *, void *, const void *, const void *, const void *, const RMSNormBackwardInfo &);

// Descriptor implementation
Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t grad_x_desc,
    infiniopTensorDescriptor_t grad_w_desc,
    infiniopTensorDescriptor_t grad_y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    
    auto info_result = RMSNormBackwardInfo::createRMSNormBackwardInfo(
        grad_x_desc, grad_w_desc, grad_y_desc, x_desc, w_desc, epsilon);
    if (!info_result) {
        return info_result.status();
    }
    
    auto cpu_handle = reinterpret_cast<device::cpu::Handle *>(handle);
    
    *desc_ptr = new Descriptor(
        nullptr,
        std::move(info_result.take()),
        0,
        cpu_handle->device,
        cpu_handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *grad_x,
    void *grad_w,
    const void *grad_y,
    const void *x,
    const void *w,
    void *stream) const {
    
    (void)workspace;
    (void)workspace_size;
    (void)stream;
    
    switch (_info.grad_x_dtype) {
        case INFINI_DTYPE_F32:
            return rmsNormBackwardImpl<float>(
                grad_x, grad_w, grad_y, x, w, _info);
        case INFINI_DTYPE_F16:
            return rmsNormBackwardImpl<fp16_t>(
                grad_x, grad_w, grad_y, x, w, _info);
        case INFINI_DTYPE_BF16:
            return rmsNormBackwardImpl<bf16_t>(
                grad_x, grad_w, grad_y, x, w, _info);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::rms_norm_backward::cpu