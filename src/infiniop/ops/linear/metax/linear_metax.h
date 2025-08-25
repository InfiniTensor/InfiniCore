#ifndef __LINEAR_METAX_H__
#define __LINEAR_METAX_H__

#include "../../../operator.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include <vector>

namespace op::linear::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    std::vector<int> _x_dims;
    std::vector<int> _w_dims;
    std::vector<int> _y_dims;
    std::vector<int> _b_dims;
    int _batch_size;
    int _in_features;
    int _out_features;
    bool _has_bias;

public:
    Descriptor() = default;
    Descriptor(infiniDtype_t dtype,
               const std::vector<int> &x_dims,
               const std::vector<int> &w_dims,
               const std::vector<int> &y_dims,
               const std::vector<int> &b_dims,
               int batch_size,
               int in_features,
               int out_features,
               bool has_bias,
               infiniDevice_t device,
               int device_id)
        : InfiniopDescriptor{device, device_id},
          _dtype(dtype), _x_dims(x_dims), _w_dims(w_dims),
          _y_dims(y_dims), _b_dims(b_dims),
          _batch_size(batch_size), _in_features(in_features),
          _out_features(out_features), _has_bias(has_bias) {}
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
    template <typename T>
    infiniStatus_t linearMetax(
        void *y_data,
        const void *x_data,
        const void *w_data,
        const void *b_data,
        void *stream) const;
};

} // namespace op::linear::metax

#endif // __LINEAR_METAX_H__