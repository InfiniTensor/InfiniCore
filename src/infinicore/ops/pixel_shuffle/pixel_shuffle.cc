#include "infinicore/ops/pixel_shuffle.hpp"
#include <numeric>

namespace infinicore::op {

Tensor pixel_shuffle(Tensor input, int64_t upscale_factor) {
    auto shape = input->shape();
    auto ndim = input->ndim();

    // if (ndim < 3) {
    //     throw std::runtime_error("pixel_shuffle: input must have at least 3 dimensions");
    // }

    size_t c_dim = ndim - 3;
    size_t h_dim = ndim - 2;
    size_t w_dim = ndim - 1;

    size_t c_r2 = shape[c_dim];
    size_t h = shape[h_dim];
    size_t w = shape[w_dim];
    size_t r2 = upscale_factor * upscale_factor;

    // if (c_r2 % r2 != 0) {
    //     throw std::runtime_error("pixel_shuffle: number of input channels must be divisible by upscale_factor^2");
    // }

    size_t c = c_r2 / r2;

    // 1. Reshape: [..., C*r^2, H, W] -> [..., C, r, r, H, W]
    Shape new_shape(shape.begin(), shape.begin() + c_dim);
    new_shape.push_back(c);
    new_shape.push_back(upscale_factor);
    new_shape.push_back(upscale_factor);
    new_shape.push_back(h);
    new_shape.push_back(w);

    Tensor x = input->view(new_shape);

    // 2. Permute: [..., C, r, r, H, W] -> [..., C, H, r, W, r]
    // Indices:
    // batch: 0 ... c_dim-1
    // C:     c_dim
    // r1:    c_dim + 1
    // r2:    c_dim + 2
    // H:     c_dim + 3
    // W:     c_dim + 4
    //
    // Target: batch, C, H, r1, W, r2
    Shape permute_order(c_dim);
    std::iota(permute_order.begin(), permute_order.end(), 0); // Fill batch dims 0, 1, ...
    
    permute_order.push_back(c_dim);     // C
    permute_order.push_back(c_dim + 3); // H
    permute_order.push_back(c_dim + 1); // r1
    permute_order.push_back(c_dim + 4); // W
    permute_order.push_back(c_dim + 2); // r2

    x = x->permute(permute_order);

    // 3. Contiguous: Copy logic happens here
    x = x->contiguous();

    // 4. Reshape: [..., C, H, r, W, r] -> [..., C, H*r, W*r]
    Shape output_shape(shape.begin(), shape.begin() + c_dim);
    output_shape.push_back(c);
    output_shape.push_back(h * upscale_factor);
    output_shape.push_back(w * upscale_factor);

    return x->view(output_shape);
}

} // namespace infinicore::op

