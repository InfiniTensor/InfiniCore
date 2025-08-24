#ifndef __LINEAR_BACKWARD_METAX_H__
#define __LINEAR_BACKWARD_METAX_H__

#include "../../../operator.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include <vector>

namespace op::linear_backward::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    std::vector<int> _grad_y_dims;
    std::vector<int> _x_dims;
    std::vector<int> _w_dims;
    int _batch_size;
    int _in_features;
    int _out_features;
    bool _compute_grad_x;
    bool _compute_grad_w;
    bool _compute_grad_b;

public:
    Descriptor() = default;
    Descriptor(infiniDtype_t dtype,
               const std::vector<int> &grad_y_dims,
               const std::vector<int> &x_dims,
               const std::vector<int> &w_dims,
               int batch_size,
               int in_features,
               int out_features,
               bool compute_grad_x,
               bool compute_grad_w,
               bool compute_grad_b,
               infiniDevice_t device,
               int device_id)
        : InfiniopDescriptor{device, device_id},
          _dtype(dtype), _grad_y_dims(grad_y_dims),
          _x_dims(x_dims), _w_dims(w_dims),
          _batch_size(batch_size), _in_features(in_features),
          _out_features(out_features), _compute_grad_x(compute_grad_x),
          _compute_grad_w(compute_grad_w), _compute_grad_b(compute_grad_b) {}
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
    template <typename T>
    infiniStatus_t linearBackwardMetax(
        void *grad_x_data,
        void *grad_w_data,
        void *grad_b_data,
        const void *grad_y_data,
        const void *x_data,
        const void *w_data,
        void *stream) const;
};

} // namespace op::linear_backward::metax

#endif // __LINEAR_BACKWARD_METAX_H__