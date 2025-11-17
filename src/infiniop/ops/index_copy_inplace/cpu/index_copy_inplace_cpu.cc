#include "index_copy_inplace_cpu.h"
#include "../../../devices/cpu/common_cpu.h"//引入CPU通用工具
#include <vector>
#include <iostream>

namespace op::index_copy_inplace::cpu {

Descriptor::~Descriptor() = default;//Descriptor的析构函数

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int dim,
    infiniopTensorDescriptor_t index_desc) {
    
    // 【追踪器 1】
    printf("--- [DEBUG] Entering Descriptor::create ---\n");

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);//将通用的 handle 转换成 CPU 专用的 handle

    //auto info = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc);
    //待定义,在.h文件中定义IndexCopyInplaceInfo类，类内定义createIndexCopyInplaceInfo函数
    auto info = IndexCopyInplaceInfo::createIndexCopyInplaceInfo(input_desc, output_desc, dim, index_desc);
    CHECK_RESULT(info);

    // Create descriptor
    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);
    // 【追踪器 】
    printf("--- [DEBUG] Exiting Descriptor::create ---\n");

    return INFINI_STATUS_SUCCESS;
}

//创建模板化的，真正的计算内核
//template <typename Tdata, typename Tindex>，这里只需要一个模板参数T，因为索引类型是固定的
template <typename Tdata>
infiniStatus_t calculateIndexCopyInplace(const IndexCopyInplaceInfo &info,
                             const Tdata *input_data,
                             Tdata *output_data,//????dim在info对象中，这里不用显示定义
                             const int64_t *index_data) {
    // a. 将 void* 指针安全地转换成具体类型的指针，这部分rope对应代码中没有！！！这里报错，删掉
    // auto output_ptr = reinterpret_cast<T *>(output_data);
    // auto input_ptr = reinterpret_cast<const T *>(input_data);
    // auto index_ptr = reinterpret_cast<const int64_t *>(index_data);

    // 【追踪器 3】
    printf("--- [DEBUG] Entering calculateKernel. slice_size = %ld, index_size = %zu\n", info.slice_size, info.index_size);
    fflush(stdout); // 强制刷新缓冲区，确保我们能立刻看到输出

    if (info.slice_size == 0 && info.output_shape.size() > 0) return INFINI_STATUS_SUCCESS;// 只有在非0维空张量时才返回

//#pragma omp parallel for
    //遍历除了dim之外的所有元素组合
    for(int64_t slice_idx = 0; slice_idx < info.slice_size; ++slice_idx){

        // 【追踪器 4】- 这个可能会打印很多次，如果卡住，我们可能看不到它
        if (slice_idx % 100 == 0) { // 每 100 次迭代打印一次，防止刷屏
             printf("--- [DEBUG] In calculateKernel loop, slice_idx = %ld\n", slice_idx);
             fflush(stdout);
        }

        int64_t output_slice_offset = 0;
        int64_t input_slice_offset = 0;
        // int64_t temp_slice_idx = slice_idx;
        // ptrdiff_t num_dims = info.output_shape.size();

        //通过stride计算每个slice的基地址偏移量
        //这是支持任意布局的关键
        // 【修正】当维度>0时，才进行地址计算
        //if (info.output_shape.size() > 0)
        if (!info.output_shape.empty()) {
            int64_t temp_slice_idx = slice_idx;
            ptrdiff_t num_dims = info.output_shape.size();

            // 【修正】使用绝对安全的倒序循环
            for (ptrdiff_t i = num_dims - 1; i >= 0; --i) {
                if (i == info.dim) continue;
                
                size_t current_dim_idx = temp_slice_idx % info.output_shape[i];
                temp_slice_idx /= info.output_shape[i];
                
                output_slice_offset += current_dim_idx * info.output_strides[i];
                input_slice_offset += current_dim_idx * info.input_strides[i];
            }
        }
        Tdata *output_slice_ptr = output_data + output_slice_offset;
        const Tdata *input_slice_ptr = input_data + input_slice_offset;

        //在当前slice上，根据index张量进行复制
        for(size_t i = 0; i < info.index_size; ++i){
            int64_t target_idx = index_data[i];
            //边界检查，防止非法内存访问
            // 0 维张量的 shape[0] 会越界
            if (info.output_shape.empty()) { // 单独处理 0 维张量
                 if (target_idx == 0) { // 索引必须是0
                     *output_slice_ptr = *input_slice_ptr;
                 }
            } else {
                if (target_idx >= 0 && static_cast<size_t>(target_idx) < info.output_shape[info.dim]) {
                    output_slice_ptr[target_idx * info.output_strides[info.dim]] =
                        input_slice_ptr[i * info.input_strides[info.dim]];
                }
            }
        }
    }
    // 【追踪器 5】
    printf("--- [DEBUG] Exiting calculateKernel ---\n");
    fflush(stdout);

    return INFINI_STATUS_SUCCESS;
}

 #define CALCULATE_INDEXCOPYINPLACE(TDATA) \
     calculateIndexCopyInplace<TDATA>(_info, \
                            static_cast<const TDATA *>(input), \
                            static_cast<TDATA *>(output), /*这里也显示转换*/ \
                           static_cast<const int64_t *>(index))
infiniStatus_t Descriptor::calculate(
    const void *input,
    void *output,
    const void *index,
    void *stream) const {
    // 【追踪器 2】
    printf("--- [DEBUG] Entering Descriptor::calculate. DType = %d\n", _info.data_type);
    fflush(stdout);

    switch (_info.data_type) {//！！！！！data_type这个命名和.h文件中的类相关，可以后面再改
    case INFINI_DTYPE_F16:
        // return CALCULATE_KERNEL<fp16_t>(_info, output, input, index);
        return CALCULATE_INDEXCOPYINPLACE(fp16_t);
    case INFINI_DTYPE_BF16:
        // return CALCULATE_KERNEL<bf16_t>(_info, output, input, index);
        return CALCULATE_INDEXCOPYINPLACE(bf16_t);
    case INFINI_DTYPE_F32:
        // return CALCULATE_KERNEL<float>(_info, output, input, index);
        return CALCULATE_INDEXCOPYINPLACE(float);
    case INFINI_DTYPE_F64:
        // return CALCULATE_KERNEL<double>(_info, output, input, index);
        return CALCULATE_INDEXCOPYINPLACE(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

//#undef ROPE_TYPE这里没有定义，所以不用取消定义宏
#undef CALCULATE_INDEXCOPYINPLACE

} // namespace op::index_copy_inplace::cpu

//---------------------------空骨架测试用-----------------------------------------
// #include <vector>
// #include <iostream>
// #include "infiniop/index_copy_inplace.h"
// #include "../../../tensor.h" // 引入 InfiniopTensorDescriptor 结构体
// #include "../../../utils.h"

// // 我们不再使用复杂的 C++ 类和宏，直接写 C 函数的实现

// // 全局函数，不再是类的成员
// template <typename Tdata>
// infiniStatus_t calculateKernel(const void *input_data, Tdata *output_data,
//                                const int64_t *index_data,
//                                const InfiniopTensorDescriptor* output_desc,
//                                const InfiniopTensorDescriptor* input_desc,
//                                const InfiniopTensorDescriptor* index_desc,
//                                int dim) {
//     // 这个函数体可以暂时为空，我们先测试链接
//     printf("--- [DEBUG] In calculateKernel! ---\n");
//     fflush(stdout);
//     return INFINI_STATUS_SUCCESS;
// }

// // extern "C" 确保这些是纯 C 风格的函数，避免名称混淆
// extern "C" {

// infiniStatus_t infiniopCreateIndexCopyInplaceDescriptor(
//     infiniopHandle_t handle,
//     infiniopIndexCopyInplaceDescriptor_t *desc_ptr,
//     infiniopTensorDescriptor_t input,
//     infiniopTensorDescriptor_t output,
//     int dim,
//     infiniopTensorDescriptor_t index) {
    
//     printf("--- [DEBUG] C-API: CreateDescriptor called ---\n");
//     fflush(stdout);
//     // 我们暂时不创建任何复杂的 C++ 对象，只返回一个虚拟的指针
//     // 这里的 42 是一个魔数，只要它不是 nullptr 即可
//     *desc_ptr = reinterpret_cast<infiniopIndexCopyInplaceDescriptor_t>(42); 
//     return INFINI_STATUS_SUCCESS;
// }

// infiniStatus_t infiniopIndexCopyInplace(
//     infiniopIndexCopyInplaceDescriptor_t desc,
//     const void *input,
//     void *output,
//     const void *index,
//     void *stream) {
    
//     printf("--- [DEBUG] C-API: IndexCopyInplace called ---\n");
//     fflush(stdout);
//     // 我们暂时不调用 kernel，只返回成功
//     return INFINI_STATUS_SUCCESS;
// }

// infiniStatus_t infiniopDestroyIndexCopyInplaceDescriptor(
//     infiniopIndexCopyInplaceDescriptor_t desc) {
//     printf("--- [DEBUG] C-API: DestroyDescriptor called ---\n");
//     fflush(stdout);
//     return INFINI_STATUS_SUCCESS;
// }

// } // extern "C"
//----------------------------抛弃模仿 rope 的、复杂的面向对象封装----------------------------------------
// #include <vector>
// #include <iostream>
// #include "infiniop/index_copy_inplace.h" // 包含 C-API 声明
// #include "../../../tensor.h"             // 引入 InfiniopTensorDescriptor 结构体
// #include "../../../../utils.h"              // 引入 CHECK_... 宏

// // 1. 将 Info 结构体直接定义在 .cc 文件内部，作为一个私有辅助工具
// struct IndexCopyInplaceInfo {
//     // Info 类的成员变量
//     infiniDtype_t data_type;
//     int dim;
//     std::vector<size_t> output_shape;
//     std::vector<ptrdiff_t> output_strides;
//     std::vector<size_t> input_shape;
//     std::vector<ptrdiff_t> input_strides;
//     size_t index_size;
//     int64_t slice_size;

//     // Info 类的 create 方法，负责所有验证
//     static utils::Result<IndexCopyInplaceInfo> create(
//         const infiniopTensorDescriptor_t input_desc,
//         const infiniopTensorDescriptor_t output_desc,
//         int dim_val,
//         const infiniopTensorDescriptor_t index_desc) {
        
//         CHECK_OR_RETURN(
//             input_desc != nullptr && output_desc != nullptr && index_desc != nullptr,
//             INFINI_STATUS_NULL_POINTER);

//         const infiniDtype_t dtype = output_desc->dtype();

//         CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);
//         CHECK_OR_RETURN(dtype == input_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
//         CHECK_OR_RETURN(index_desc->dtype() == INFINI_DTYPE_I64, INFINI_STATUS_BAD_TENSOR_DTYPE);

//         CHECK_OR_RETURN(output_desc->ndim() == input_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
//         CHECK_OR_RETURN(index_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        
//         if (output_desc->ndim() == 0) {
//             CHECK_OR_RETURN(dim_val == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
//         } else {
//             CHECK_OR_RETURN(dim_val >= 0 && static_cast<size_t>(dim_val) < output_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
//         }
        
//         for (size_t i = 0; i < output_desc->ndim(); ++i) {
//             if (i != static_cast<size_t>(dim_val)) {
//                 CHECK_OR_RETURN(output_desc->dim(i) == input_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
//             }
//         }
//         if (output_desc->ndim() > 0) {
//             CHECK_OR_RETURN(input_desc->dim(dim_val) == index_desc->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);
//         }

//         int64_t current_slice_size = 1;
//         if (output_desc->ndim() > 0) {
//             for (size_t i = 0; i < output_desc->ndim(); ++i) {
//                 if (i != static_cast<size_t>(dim_val)) {
//                     current_slice_size *= output_desc->dim(i);
//                 }
//             }
//         }
        
//         return utils::Result<IndexCopyInplaceInfo>(IndexCopyInplaceInfo{
//             dtype, dim_val, output_desc->shape(), output_desc->strides(),
//             input_desc->shape(), input_desc->strides(),
//             index_desc->numel(), current_slice_size,
//         });
//     }
// };

// // 2. 真正的计算内核函数
// template <typename Tdata>
// infiniStatus_t calculateKernel(const IndexCopyInplaceInfo &info,
//                                const Tdata *input_data,
//                                Tdata *output_data,
//                                const int64_t *index_data) {
    
//     if (info.slice_size == 0 && !info.output_shape.empty()) return INFINI_STATUS_SUCCESS;

//     // #pragma omp parallel for // 先在单线程模式下验证正确性
//     for (int64_t slice_idx = 0; slice_idx < info.slice_size; ++slice_idx) {
//         int64_t output_slice_offset = 0;
//         int64_t input_slice_offset = 0;
        
//         if (!info.output_shape.empty()) {
//             int64_t temp_slice_idx = slice_idx;
//             ptrdiff_t num_dims = info.output_shape.size();

//             for (ptrdiff_t i = num_dims - 1; i >= 0; --i) {
//                 if (i == info.dim) continue;
//                 size_t current_dim_idx = temp_slice_idx % info.output_shape[i];
//                 temp_slice_idx /= info.output_shape[i];
//                 output_slice_offset += current_dim_idx * info.output_strides[i];
//                 input_slice_offset += current_dim_idx * info.input_strides[i];
//             }
//         }

//         Tdata *output_slice_ptr = output_data + output_slice_offset;
//         const Tdata *input_slice_ptr = input_data + input_slice_offset;

//         for (size_t i = 0; i < info.index_size; ++i) {
//             int64_t target_idx = index_data[i];
            
//             if (info.output_shape.empty()) {
//                  if (target_idx == 0) { *output_slice_ptr = *input_slice_ptr; }
//             } else {
//                 if (target_idx >= 0 && static_cast<size_t>(target_idx) < info.output_shape[info.dim]) {
//                     output_slice_ptr[target_idx * info.output_strides[info.dim]] =
//                         input_slice_ptr[i * info.input_strides[info.dim]];
//                 }
//             }
//         }
//     }
//     return INFINI_STATUS_SUCCESS;
// }

// // 3. 实现 C-API 函数
// extern "C" {

// infiniStatus_t infiniopCreateIndexCopyInplaceDescriptor(
//     infiniopHandle_t handle,
//     infiniopIndexCopyInplaceDescriptor_t *desc_ptr,
//     infiniopTensorDescriptor_t input,
//     infiniopTensorDescriptor_t output,
//     int dim,
//     infiniopTensorDescriptor_t index) {
    
//     auto info_result = IndexCopyInplaceInfo::create(input, output, dim, index);
//     CHECK_RESULT(info_result);
    
//     auto info = new IndexCopyInplaceInfo(info_result.take());
//     *desc_ptr = reinterpret_cast<infiniopIndexCopyInplaceDescriptor_t>(info);
    
//     return INFINI_STATUS_SUCCESS;
// }

// infiniStatus_t infiniopIndexCopyInplace(
//     infiniopIndexCopyInplaceDescriptor_t desc,
//     const void *input, void *output, const void *index, void *stream) {
    
//     auto info = reinterpret_cast<const IndexCopyInplaceInfo *>(desc);

//     switch (info->data_type) {
//         case INFINI_DTYPE_F16:
//             return calculateKernel<fp16_t>(*info, reinterpret_cast<const fp16_t*>(input), reinterpret_cast<fp16_t*>(output), reinterpret_cast<const int64_t*>(index));
//         case INFINI_DTYPE_F32:
//             return calculateKernel<float>(*info, reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output), reinterpret_cast<const int64_t*>(index));
//         case INFINI_DTYPE_BF16:
//             return calculateKernel<bf16_t>(*info, reinterpret_cast<const bf16_t*>(input), reinterpret_cast<bf16_t*>(output), reinterpret_cast<const int64_t*>(index));
//         case INFINI_DTYPE_F64:
//             return calculateKernel<double>(*info, reinterpret_cast<const double*>(input), reinterpret_cast<double*>(output), reinterpret_cast<const int64_t*>(index));
//         default:
//             return INFINI_STATUS_BAD_TENSOR_DTYPE;
//     }
// }

// infiniStatus_t infiniopDestroyIndexCopyInplaceDescriptor(
//     infiniopIndexCopyInplaceDescriptor_t desc) {
//     delete reinterpret_cast<const IndexCopyInplaceInfo *>(desc);
//     return INFINI_STATUS_SUCCESS;
// }

// } // extern "C"
//----------------------------给operator.cc转移主要功能----------------------------------------
// #include "index_copy_inplace_cpu.h"
// #include <vector>

// // 模板化的 Kernel 函数
// template <typename Tdata>
// infiniStatus_t calculateKernel(const IndexCopyInplaceInfo &info,
//                                const Tdata *input_data,
//                                Tdata *output_data,
//                                const int64_t *index_data) {
    
//     if (info.slice_size == 0 && !info.output_shape.empty()) return INFINI_STATUS_SUCCESS;

//     // #pragma omp parallel for // 先在单线程模式下验证正确性
//     for (int64_t slice_idx = 0; slice_idx < info.slice_size; ++slice_idx) {
//         int64_t output_slice_offset = 0;
//         int64_t input_slice_offset = 0;
        
//         if (!info.output_shape.empty()) {
//             int64_t temp_slice_idx = slice_idx;
//             ptrdiff_t num_dims = info.output_shape.size();

//             for (ptrdiff_t i = num_dims - 1; i >= 0; --i) {
//                 if (i == info.dim) continue;
//                 size_t current_dim_idx = temp_slice_idx % info.output_shape[i];
//                 temp_slice_idx /= info.output_shape[i];
//                 output_slice_offset += current_dim_idx * info.output_strides[i];
//                 input_slice_offset += current_dim_idx * info.input_strides[i];
//             }
//         }

//         Tdata *output_slice_ptr = output_data + output_slice_offset;
//         const Tdata *input_slice_ptr = input_data + input_slice_offset;

//         for (size_t i = 0; i < info.index_size; ++i) {
//             int64_t target_idx = index_data[i];
            
//             if (info.output_shape.empty()) {
//                  if (target_idx == 0) { *output_slice_ptr = *input_slice_ptr; }
//             } else {
//                 if (target_idx >= 0 && static_cast<size_t>(target_idx) < info.output_shape[info.dim]) {
//                     output_slice_ptr[target_idx * info.output_strides[info.dim]] =
//                         input_slice_ptr[i * info.input_strides[info.dim]];
//                 }
//             }
//         }
//     }
//     return INFINI_STATUS_SUCCESS;
// }

// // CPU 专属的内核启动器
// infiniStatus_t index_copy_inplace_kernel_cpu(
//     const IndexCopyInplaceInfo &info,
//     const void *input, void *output, const void *index, void *stream) {
    
//     switch (info.data_type) {
//         case INFINI_DTYPE_F16:
//             return calculateKernel<fp16_t>(info, reinterpret_cast<const fp16_t*>(input), reinterpret_cast<fp16_t*>(output), reinterpret_cast<const int64_t*>(index));
//         case INFINI_DTYPE_F32:
//             return calculateKernel<float>(info, reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output), reinterpret_cast<const int64_t*>(index));
//         case INFINI_DTYPE_BF16:
//             return calculateKernel<bf16_t>(info, reinterpret_cast<const bf16_t*>(input), reinterpret_cast<bf16_t*>(output), reinterpret_cast<const int64_t*>(index));
//         case INFINI_DTYPE_F64:
//             return calculateKernel<double>(info, reinterpret_cast<const double*>(input), reinterpret_cast<double*>(output), reinterpret_cast<const int64_t*>(index));
//         default:
//             return INFINI_STATUS_BAD_TENSOR_DTYPE;
//     }
//     return INFINI_STATUS_BAD_TENSOR_DTYPE;
// }