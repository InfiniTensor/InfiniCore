#ifndef __LEAKY_RELU_CPU_H__
#define __LEAKY_RELU_CPU_H__

#include <vector>

// 引入elementwise（自动定义Descriptor）的宏
#include "../../../elementwise/cpu/elementwise_cpu.h"

// 取消自动定义Descriptor的宏函数
// ELEMENTWISE_DESCRIPTOR(leaky_relu, cpu)

namespace op::leaky_relu::cpu {                                             
    class Descriptor final : public InfiniopDescriptor {                      
        infiniDtype_t _dtype;                                                 
        op::elementwise::ElementwiseInfo _info;                               
        std::unique_ptr<op::elementwise::cpu::DeviceImpl> _device_info; 
        size_t _workspace_size;                                               
                                                                              
        Descriptor(                                                           
            infiniDtype_t dtype,                                              
            op::elementwise::ElementwiseInfo info,                            
            op::elementwise::cpu::DeviceImpl *device_info,              
            size_t workspace_size,                                            
            infiniDevice_t device_type,                                       
            int device_id)                                                    
            : InfiniopDescriptor{device_type, device_id},                     
              _dtype(dtype),                                                  
              _info(std::move(info)),                                         
              _device_info(std::move(device_info)),                           
              _workspace_size(workspace_size) {}                              
                                                                              
    public:                                                                   
        ~Descriptor();                                                        
                                                                              
        size_t workspaceSize() const { return _workspace_size; }              
                                                                              
        static infiniStatus_t create(                                         
            infiniopHandle_t handle,                                          
            Descriptor **desc_ptr,                                            
            infiniopTensorDescriptor_t output_desc,                           
            std::vector<infiniopTensorDescriptor_t> input_descs);             
                                                                              
        infiniStatus_t calculate(                                             
            void *workspace, size_t workspace_size,                           
            void *output,                                                     
            std::vector<const void *> inputs,                   
            float negative_slope,              
            void *stream) const;                                              
    };                                                                        
    }

namespace op::leaky_relu::cpu {
typedef struct LeakyReluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x, float negative_slope) const {
        return x >= T(0) ? x : negative_slope * x;
    }
} LeakyReluOp;
} // namespace op::leaky_relu::cpu

#endif // __LEAKY_RELU_CPU_H__
