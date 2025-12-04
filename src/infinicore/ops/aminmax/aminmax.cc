#include "infinicore/ops/aminmax.hpp"
#include "infinicore/context/context.hpp"
#include <utility>

namespace infinicore::op {

common::OpDispatcher<Aminmax::schema> &Aminmax::dispatcher() {
    static common::OpDispatcher<Aminmax::schema> dispatcher_;
    return dispatcher_;
}

void Aminmax::execute(Tensor min_output, Tensor max_output, Tensor input,
                      std::optional<int64_t> dim, bool keepdim) {
    dispatcher().lookup(context::getDevice().getType())(
        min_output, max_output, input, dim, keepdim);
}

// 计算输出形状（与 any 类似）
static Shape compute_output_shape(const Shape &input_shape,
                                  std::optional<int64_t> dim,
                                  bool keepdim) {
    if (!dim.has_value()) {
        // 全局 reduce
        if (keepdim) {
            return Shape(input_shape.size(), 1);
        } else {
            return Shape{}; // 标量
        }
    } else {
        int64_t d = dim.value();
        if (d < 0) {
            d += static_cast<int64_t>(input_shape.size());
        }
        Shape output_shape = input_shape;
        if (keepdim) {
            output_shape[d] = 1;
        } else {
            output_shape.erase(output_shape.begin() + d);
        }
        return output_shape;
    }
}

std::pair<Tensor, Tensor> aminmax(Tensor input,
                                  std::optional<int64_t> dim,
                                  bool keepdim) {
    auto output_shape = compute_output_shape(input->shape(), dim, keepdim);
    auto min_output = Tensor::empty(output_shape, input->dtype(), input->device());
    auto max_output = Tensor::empty(output_shape, input->dtype(), input->device());
    aminmax_(min_output, max_output, input, dim, keepdim);
    return {min_output, max_output};
}

void aminmax_(Tensor min_output, Tensor max_output, Tensor input,
              std::optional<int64_t> dim, bool keepdim) {
    Aminmax::execute(min_output, max_output, input, dim, keepdim);
}

} // namespace infinicore::op
