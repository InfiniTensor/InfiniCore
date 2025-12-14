#include "infinicore/ops/pixel_shuffle.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<PixelShuffle::schema> &PixelShuffle::dispatcher() {
    static common::OpDispatcher<PixelShuffle::schema> dispatcher_;
    return dispatcher_;
};

void PixelShuffle::execute(Tensor output, Tensor input, int64_t upscale_factor) {
    dispatcher().lookup(context::getDevice().getType())(output, input, upscale_factor);
}

Tensor pixel_shuffle(Tensor input, int64_t upscale_factor) {
    Shape input_shape = input->shape();
    Size ndim = input->ndim();
    
    if (ndim < 3) {
        throw std::runtime_error("pixel_shuffle: input must have at least 3 dimensions");
    }
    
    // Input shape: (*, C*r^2, H, W)
    // Output shape: (*, C, H*r, W*r)
    Size c_r2 = input_shape[ndim - 3];
    Size h = input_shape[ndim - 2];
    Size w = input_shape[ndim - 1];
    
    if (c_r2 % (upscale_factor * upscale_factor) != 0) {
        throw std::runtime_error("pixel_shuffle: number of input channels must be divisible by upscale_factor^2");
    }
    
    Size c = c_r2 / (upscale_factor * upscale_factor);
    Size output_h = h * upscale_factor;
    Size output_w = w * upscale_factor;
    
    // Calculate output shape
    Shape output_shape = input_shape;
    output_shape[ndim - 3] = c;
    output_shape[ndim - 2] = output_h;
    output_shape[ndim - 1] = output_w;
    
    // Convert input to contiguous if needed to avoid issues with non-contiguous strides
    Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();
    
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    pixel_shuffle_(output, input_contiguous, upscale_factor);
    return output;
}

void pixel_shuffle_(Tensor output, Tensor input, int64_t upscale_factor) {
    PixelShuffle::execute(output, input, upscale_factor);
}

} // namespace infinicore::op

