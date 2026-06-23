#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <cstddef>
#include <optional>

namespace infinicore::op {
class Conv1d {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor,
                            const size_t *, const size_t *, const size_t *, size_t);
    static void execute(Tensor output,
                        Tensor input,
                        Tensor weight,
                        Tensor bias,
                        const size_t *pads,
                        const size_t *strides,
                        const size_t *dilations,
                        size_t n);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor conv1d(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias,
              size_t stride,
              size_t padding,
              size_t dilation,
              size_t groups);
void conv1d_(Tensor output,
             Tensor input,
             Tensor weight,
             std::optional<Tensor> bias,
             size_t stride,
             size_t padding,
             size_t dilation,
             size_t groups);
} // namespace infinicore::op
