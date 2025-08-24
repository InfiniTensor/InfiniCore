#ifndef __LINEAR_BACKWARD_CPU_H__
#define __LINEAR_BACKWARD_CPU_H__

#include <vector>
#include "../../../operator.h"
#include "../../../devices/cpu/cpu_handle.h"

namespace op::linear_backward::cpu {

class Descriptor : public InfiniopDescriptor {
public:
    Descriptor() = default;
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t grad_y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t grad_x_desc,
        infiniopTensorDescriptor_t grad_w_desc,
        infiniopTensorDescriptor_t grad_b_desc);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *grad_x,
        void *grad_w,
        void *grad_b,
        const void *grad_y,
        const void *x,
        const void *w,
        void *stream) const;

private:
    infiniopTensorDescriptor_t _grad_y_desc;
    infiniopTensorDescriptor_t _x_desc;
    infiniopTensorDescriptor_t _w_desc;
    infiniopTensorDescriptor_t _grad_x_desc;
    infiniopTensorDescriptor_t _grad_w_desc;
    infiniopTensorDescriptor_t _grad_b_desc;
    device::cpu::Handle *_handle;
    
    // Save tensor shapes and data type to avoid accessing descriptors later
    std::vector<int> _grad_y_dims;
    std::vector<int> _x_dims;
    std::vector<int> _w_dims;
    infiniDtype_t _dtype;
};

} // namespace op::linear_backward::cpu

#endif // __LINEAR_BACKWARD_CPU_H__