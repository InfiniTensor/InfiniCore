#ifndef ADD_H
#define ADD_H

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/add.h"
#include "../../../ninetoothed/utils.h"

namespace op::add::ninetoothed {
class Descriptor final : public InfiniopDescriptor {

public:
    Descriptor(
        infiniopHandle_t handle,
        infiniopTensorDescriptor_t c_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) : InfiniopDescriptor{handle->device, handle->device_id},
                                                                  c_shape_{c_desc->shape()},
                                                                  c_strides_{c_desc->strides()},
                                                                  a_shape_{input_desc_vec[0]->shape()},
                                                                  a_strides_{input_desc_vec[0]->strides()},
                                                                  b_shape_{input_desc_vec[1]->shape()},
                                                                  b_strides_{input_desc_vec[1]->strides()},
                                                                  dtype_{c_desc->dtype()} {}

    ~Descriptor() = default;

    size_t workspaceSize() const {
        return 0;
    }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec) {
        *desc_ptr = new Descriptor(handle, c_desc, input_desc_vec);
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        std::vector<const void *> inputs,
        void *stream) const {
        auto c_nt{::ninetoothed::Tensor(c, c_shape_, c_strides_)};
        auto a_nt{::ninetoothed::Tensor(inputs[0], a_shape_, a_strides_)};
        auto b_nt{::ninetoothed::Tensor(inputs[1], b_shape_, b_strides_)};

        if (launch_add(stream,
                          c_nt,
                          a_nt,
                          b_nt,
                          c_shape_.size(),
                          dtype_,
                          1024)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }

        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    std::vector<Size> c_shape_;
    std::vector<Stride> c_strides_;

    std::vector<Size> a_shape_;
    std::vector<Stride> a_strides_;

    std::vector<Size> b_shape_;
    std::vector<Stride> b_strides_;

    infiniDtype_t dtype_;
};
} // namespace op::add::ninetoothed

#endif // ADD_H
