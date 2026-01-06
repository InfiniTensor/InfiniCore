#ifndef __HARDTANH_CUDA_API_H__
#define __HARDTANH_CUDA_API_H__

#include "../../../elementwise/nvidia/elementwise_nvidia_api.cuh"

// 1. 使用宏注册 hardtanh 算子在 nvidia 后端的描述符基础结构
ELEMENTWISE_DESCRIPTOR(hardtanh, nvidia)

namespace op::hardtanh::nvidia {

// 2. 显式定义 Descriptor 类以存储算子特有的参数
// 注意：该类是由上面的 ELEMENTWISE_DESCRIPTOR 宏生成的模板或基类的特化
class Descriptor : public elementwise::nvidia::Descriptor {
public:
    // 存储 HardTanh 截断范围
    float min_val;
    float max_val;

    // 析构函数（对应 .cu 中的实现）
    ~Descriptor() override;

    // 静态创建函数（对应 .cu 中的实现）
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec,
        float min_val,
        float max_val);

    // 计算执行函数（对应 .cu 中的实现）
    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const override;
};

} // namespace op::hardtanh::nvidia

#endif // __HARDTANH_CUDA_API_H__