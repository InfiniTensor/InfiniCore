#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <pybind11/pybind11.h>
namespace infinicore::op {
class Attention {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, size_t);
    static void execute(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor attention(Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos);
void attention_(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos);

Tensor scaled_dot_product_attention(Tensor query,
                                    Tensor key,
                                    Tensor value,
                                    pybind11::object scale);

void scaled_dot_product_attention_(Tensor out,
                                   Tensor query,
                                   Tensor key,
                                   Tensor value,
                                   pybind11::object scale);
} // namespace infinicore::op
