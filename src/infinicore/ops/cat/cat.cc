#include "infinicore/ops/cat.hpp"
#include "infinicore/context/context.hpp"
#include <cstring>

namespace infinicore::op {

namespace {

    void low_dim_copy(Tensor& tensor, int dim, Tensor& out, std::byte* tensor_ptr, std::byte* out_ptr, int depth) {
        if (depth != out->ndim()) {
            std::byte* now_tensor_ptr = tensor_ptr;
            std::byte* now_out_ptr = out_ptr;
            for (int i = 0; i < tensor->shape()[depth]; i ++) {
                low_dim_copy(tensor, dim, out, now_tensor_ptr, now_out_ptr, depth + 1);
                now_tensor_ptr += tensor->stride(depth) * dsize(tensor->dtype());
                now_out_ptr += out->stride(depth) * dsize(out->dtype());
            }
        } else {
            Size data_size = dsize(out->dtype());
            if (out->device().getType() == Device::Type::CPU)
                std::memcpy(out_ptr, tensor_ptr, data_size);
            else
                context::memcpyD2D(out_ptr, tensor_ptr, data_size);
        }
    }

    void high_dim_split(std::vector<Tensor>& tensors, int dim, Tensor& out, std::vector<std::byte*> tensors_ptr, std::byte* out_ptr, int depth) {
        if (depth != dim) {
            std::vector<std::byte*> now_tensors_ptr = tensors_ptr;
            std::byte* now_out_ptr = out_ptr;
            for (int i = 0; i < out->shape()[depth]; i ++) {
                high_dim_split(tensors, dim, out, now_tensors_ptr, now_out_ptr, depth + 1);
                for (int i = 0; i < tensors.size(); i ++) {
                    if (tensors[i]->ndim() == 1)    continue;
                    now_tensors_ptr[i] += tensors[i]->stride(depth) * dsize(tensors[i]->dtype());
                }
                now_out_ptr += out->stride(depth) * dsize(out->dtype());
            }
        } else {
            std::byte* now_out_ptr = out_ptr;
            for (int i = 0; i < tensors.size(); i ++) {
                if (tensors[i]->ndim() == 1)    continue;
                low_dim_copy(tensors[i], dim, out, tensors_ptr[i], now_out_ptr, depth);
                now_out_ptr += tensors[i]->shape()[depth] * out->stride(depth) * dsize(out->dtype());
            }
        }
    }

} // namespace

Tensor cat(std::vector<Tensor> tensors, int dim) {
    assert(tensors.size() >= 2);
    int ndim = tensors[0]->ndim();
    assert(-ndim <= dim && dim < ndim);
    dim = (dim + ndim) % ndim;

    Shape shape = tensors[0]->shape();
    for (int i = 1; i < tensors.size(); i ++) {
        assert(tensors[i]->ndim() == dim || tensors[i]->ndim() == 1);
        if (tensors[i]->ndim() != ndim) continue;
        shape[dim] += tensors[i]->shape()[dim];
    }

    auto out = Tensor::empty(shape, tensors[0]->dtype(), tensors[0]->device());
    cat_(tensors, dim, out);
    return out;
}

void cat_(std::vector<Tensor> tensors, int dim, Tensor out) {
    assert(tensors.size() >= 2);
    int ndim = out->ndim();
    assert(-ndim <= dim && dim < ndim);
    dim = (dim + ndim) % ndim;


    Size dim_shape = 0;
    for (auto& tensor : tensors) {
        assert(tensor->ndim() == ndim || tensors[i]->ndim() == 1);
        if (tensor->ndim() == 1) {
            assert(tensor->shape()[0] == 0);
            continue;
        }
        for (int i = 0; i < ndim; i ++) {
            if (i != dim) {
                assert(tensor->shape()[i] == out->shape()[i]);
            } else {
                dim_shape += tensor->shape()[i];
            }
        }
    }
    assert(dim_shape == out->shape()[dim]);

    std::vector<std::byte*> tensors_ptr(tensors.size());

    for (int i = 0; i < tensors.size(); i ++) {
        tensors_ptr[i] = tensors[i]->data();
    }
    std::byte* out_ptr = out->data();

    high_dim_split(tensors, dim, out, tensors_ptr, out_ptr, 0);

}

} // namespace infinicore::op