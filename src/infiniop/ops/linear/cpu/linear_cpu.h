#ifndef __LINEAR_CPU_H__
#define __LINEAR_CPU_H__

#include "../../../operator.h"
#include "../../../devices/cpu/cpu_handle.h"
#include <vector>

namespace op::linear::cpu {

class Descriptor : public InfiniopDescriptor {
public:
    Descriptor() = default;
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t y_desc);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        const void *w,
        const void *b,
        void *stream) const;

private:
    infiniopTensorDescriptor_t _x_desc;
    infiniopTensorDescriptor_t _w_desc;
    infiniopTensorDescriptor_t _b_desc;
    infiniopTensorDescriptor_t _y_desc;
    device::cpu::Handle *_handle;
    std::vector<int> _x_dims;
    std::vector<int> _w_dims;
    infiniDtype_t _dtype;
};

} // namespace op::linear::cpu

#endif // __LINEAR_CPU_H__