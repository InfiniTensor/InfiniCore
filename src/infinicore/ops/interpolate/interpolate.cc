#include "infinicore/ops/interpolate.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Interpolate);

Interpolate::Interpolate(Tensor out,
                         const Tensor &input,
                         std::string mode,
                         std::vector<int64_t> size,
                         std::vector<double> scale_factor,
                         int align_corners) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, input, std::move(mode), std::move(size), std::move(scale_factor), align_corners);
}

void Interpolate::execute(Tensor out,
                          const Tensor &input,
                          std::string mode,
                          std::vector<int64_t> size,
                          std::vector<double> scale_factor,
                          int align_corners) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Interpolate, out, input, std::move(mode), std::move(size), std::move(scale_factor), align_corners);
}

static std::vector<size_t> infer_interpolate_shape(
    const std::vector<size_t> &input_shape,
    const std::vector<int64_t> &size,
    const std::vector<double> &scale_factor) {
    if (input_shape.size() < 3) {
        throw std::runtime_error("interpolate expects input with at least 3 dimensions");
    }

    const size_t spatial_ndim = input_shape.size() - 2;
    std::vector<size_t> out_shape = input_shape;

    const bool has_size = !size.empty();
    const bool has_scale = !scale_factor.empty();
    if (has_size == has_scale) {
        throw std::runtime_error("interpolate expects exactly one of size or scale_factor");
    }

    if (has_size) {
        if (size.size() != spatial_ndim) {
            throw std::runtime_error("interpolate size dimensionality mismatch");
        }
        for (size_t i = 0; i < spatial_ndim; ++i) {
            out_shape[i + 2] = static_cast<size_t>(size[i]);
        }
        return out_shape;
    }

    const double scale = scale_factor[0];
    for (size_t i = 0; i < spatial_ndim; ++i) {
        out_shape[i + 2] = static_cast<size_t>(static_cast<double>(input_shape[i + 2]) * scale);
    }
    return out_shape;
}

Tensor interpolate(const Tensor &input,
                   std::string mode,
                   std::vector<int64_t> size,
                   std::vector<double> scale_factor,
                   int align_corners) {
    auto out_shape = infer_interpolate_shape(input->shape(), size, scale_factor);
    auto out = Tensor::empty(out_shape, input->dtype(), input->device());
    interpolate_(out, input, std::move(mode), std::move(size), std::move(scale_factor), align_corners);
    return out;
}

void interpolate_(Tensor out,
                  const Tensor &input,
                  std::string mode,
                  std::vector<int64_t> size,
                  std::vector<double> scale_factor,
                  int align_corners) {
    Interpolate::execute(out, input, std::move(mode), std::move(size), std::move(scale_factor), align_corners);
}

} // namespace infinicore::op

