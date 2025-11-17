//定义IndexCopyInplaceInfo类，类内定义createIndexCopyInplaceInfo函数
#ifndef __INFINIOP_INDEX_COPY_INPLACE_H__
#define __INFINIOP_INDEX_COPY_INPLACE_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <vector>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::index_copy_inplace::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        private:  /*不加注释也默认是private变量*/                  \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        IndexCopyInplaceInfo _info;                              \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            IndexCopyInplaceInfo info, /*私有构造函数*/           \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();  /* 析构函数在 .cc 文件中实现 */            \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
        /*静态工厂方法，供 C-API 调用*/                                                         \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t input_desc,                   \
            infiniopTensorDescriptor_t output_desc,                   \
            int dim,                                             \
            infiniopTensorDescriptor_t index_desc);                \
        /*核心计算方法*/                                                         \
        infiniStatus_t calculate(                                \
            const void *input,                                       \
            void *output,                                 \
            const void *index,                               \
            void *stream) const;                                 \
    };                                                           \
    }

class IndexCopyInplaceInfo {
private:
    IndexCopyInplaceInfo() = default;// 私有构造函数，强制外部使用静态 create 方法

public:
    infiniDtype_t data_type;// 保存所有 Kernel 计算需要的元数据
    int dim;
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> output_strides;
    std::vector<size_t> input_shape;
    std::vector<ptrdiff_t> input_strides;
    size_t index_size;
    int64_t slice_size;

    static utils::Result<IndexCopyInplaceInfo> createIndexCopyInplaceInfo(
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t output_desc,
        int dim_val,
        infiniopTensorDescriptor_t index_desc) {
            //检查所有指针是否为空
        CHECK_OR_RETURN(
            input_desc != nullptr && output_desc != nullptr && index_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t data_type = output_desc->dtype();

        //检查数据类型是否合法和匹配
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(data_type == input_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(index_desc->dtype() == INFINI_DTYPE_I64, INFINI_STATUS_BAD_TENSOR_DTYPE);

        //检查维度是否合法和匹配
        CHECK_OR_RETURN(output_desc->ndim() == input_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(index_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        //CHECK_OR_RETURN(dim_val >= 0 && dim_val < output_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        if (output_desc->ndim() == 0) {
            CHECK_OR_RETURN(dim_val == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
        } else {
            CHECK_OR_RETURN(dim_val >= 0 && static_cast<size_t>(dim_val) < output_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        }//这里强制类型转换为size_t无符号整数，原来dim_val是int有符号，有符号和无符号无法比较，前面限制了不可能为负数，这里直接类型转换
        
        //检查 Shape 是否匹配
        for (size_t i = 0; i < output_desc->ndim(); ++i) {//这里同样有符号和无符号无法比较，修改类型为size_t
            if (i != static_cast<size_t>(dim_val)) {//修正比较
                CHECK_OR_RETURN(output_desc->dim(i) == input_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
            }
        }
        CHECK_OR_RETURN(input_desc->dim(dim_val) == index_desc->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);

        //计算 slice_size
        int64_t current_slice_size = 1;
        if (output_desc->ndim() > 0) {
            for (size_t i = 0; i < output_desc->ndim(); ++i) {
                if (i != static_cast<size_t>(dim_val)) {//修正比较
                    current_slice_size *= output_desc->dim(i);
                }
            }
        } else { //处理 0 维张量的情况
            current_slice_size = 1; //标量的 slice_size 为 1
        }
        //验证通过，创建并返回 Info 对象
        return utils::Result<IndexCopyInplaceInfo>(IndexCopyInplaceInfo{
            data_type,
            dim_val,
            output_desc->shape(),
            output_desc->strides(),
            input_desc->shape(),
            input_desc->strides(),
            index_desc->numel(),
            current_slice_size,
        });
    }
};

#endif


//----------------------------空骨架测试用----------------------------------------
// #ifndef __INFINIOP_INDEX_COPY_INPLACE_H__
// #define __INFINIOP_INDEX_COPY_INPLACE_H__

// #include "../../../utils.h"
// #include "../../operator.h"
// #include "../../tensor.h"
// #include <vector>

// // DESCRIPTOR 宏保持不变...
// #define DESCRIPTOR(NAMESPACE) ...

// class IndexCopyInplaceInfo {
//   private:
//     IndexCopyInplaceInfo() = default;
//   public:
//     infiniDtype_t data_type;
//     int dim;
//     std::vector<size_t> output_shape;
//     std::vector<ptrdiff_t> output_strides;
//     std::vector<size_t> input_shape;
//     std::vector<ptrdiff_t> input_strides;
//     size_t index_size;
//     int64_t slice_size;

//     // 【核心修正】暂时移除所有验证逻辑，只返回一个空对象
//     static utils::Result<IndexCopyInplaceInfo> create(
//         infiniopTensorDescriptor_t input_desc,
//         infiniopTensorDescriptor_t output_desc,
//         int dim_val,
//         infiniopTensorDescriptor_t index_desc) {
        
//         // 我们暂时不进行任何检查，直接返回一个默认构造的对象
//         // 这将帮助我们判断问题是否出在这些 CHECK 宏内部
//         return utils::Result<IndexCopyInplaceInfo>(IndexCopyInplaceInfo{});
//     }
// };

// #endif // __INFINIOP_INDEX_COPY_INPLACE_H__

//----------------------------抛弃模仿 rope 的、复杂的面向对象封装----------------------------------------
// #ifndef __INFINIOP_INDEX_COPY_INPLACE_H__
// #define __INFINIOP_INDEX_COPY_INPLACE_H__
// #include "infiniop/handle.h"
// #include "infiniop/operator_descriptor.h"
// #include "infiniop/tensor_descriptor.h"
// #ifdef __cplusplus
// extern "C" {
// #endif
// typedef struct InfiniopDescriptor *infiniopIndexCopyInplaceDescriptor_t;
// __C __export infiniStatus_t infiniopCreateIndexCopyInplaceDescriptor(...);
// __C __export infiniStatus_t infiniopIndexCopyInplace(...);
// __C __export infiniStatus_t infiniopDestroyIndexCopyInplaceDescriptor(...);
// #ifdef __cplusplus
// }
// #endif
// #endif