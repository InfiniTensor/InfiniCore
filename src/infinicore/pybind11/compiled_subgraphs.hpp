#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/ops/inductor_segment.hpp"

namespace py = pybind11;

namespace infinicore::compiled_subgraphs {

namespace {

infinicore::op::PiecewiseInductorSegmentId parse_segment_id(const std::string &segment) {
    if (segment == "pre_attn") {
        return infinicore::op::PiecewiseInductorSegmentId::PreAttn;
    }
    if (segment == "post_attn_cg") {
        return infinicore::op::PiecewiseInductorSegmentId::PostAttnCg;
    }
    throw std::runtime_error("unknown piecewise segment: " + segment);
}

} // namespace

inline void bind(py::module &m) {
    m.def(
        "register_piecewise_inductor_package",
        [](const std::string &segment,
           py::ssize_t layer_idx,
           size_t bucket,
           const std::string &package_path,
           size_t tp_rank,
           bool layer_agnostic) {
            size_t reg_layer = layer_agnostic
                                   ? static_cast<size_t>(-1)
                                   : static_cast<size_t>(layer_idx);
            infinicore::op::inductor_segment_impl::register_package(
                parse_segment_id(segment),
                reg_layer,
                bucket,
                package_path,
                tp_rank,
                layer_agnostic);
        },
        py::arg("segment"),
        py::arg("layer_idx"),
        py::arg("bucket"),
        py::arg("package_path"),
        py::arg("tp_rank") = 0,
        py::arg("layer_agnostic") = false);

    m.def(
        "set_piecewise_inductor_lookup_tp_rank",
        [](size_t tp_rank) {
            infinicore::op::inductor_segment_impl::set_lookup_tp_rank_override(tp_rank);
        },
        py::arg("tp_rank"));

    m.def("clear_piecewise_inductor_lookup_tp_rank", []() {
        infinicore::op::inductor_segment_impl::clear_lookup_tp_rank_override();
    });

    m.def("clear_piecewise_inductor_packages", []() {
        infinicore::op::inductor_segment_impl::clear_packages();
    });

    m.def(
        "has_piecewise_inductor_package",
        [](const std::string &segment, size_t layer_idx, size_t bucket) {
            return infinicore::op::inductor_segment_impl::has_package(
                parse_segment_id(segment), layer_idx, bucket);
        },
        py::arg("segment"),
        py::arg("layer_idx"),
        py::arg("bucket"));

    m.def(
        "inductor_segment_",
        [](const infinicore::Tensor &positions,
           infinicore::Tensor &hidden_states,
           infinicore::Tensor &residual,
           infinicore::Tensor &q_rope,
           infinicore::Tensor &k_rope,
           infinicore::Tensor &v_rope,
           const std::string &segment,
           size_t layer_idx,
           size_t bucket) {
            infinicore::op::inductor_segment_(
                positions,
                hidden_states,
                residual,
                q_rope,
                k_rope,
                v_rope,
                parse_segment_id(segment),
                layer_idx,
                bucket);
        },
        py::arg("positions"),
        py::arg("hidden_states"),
        py::arg("residual"),
        py::arg("q_rope"),
        py::arg("k_rope"),
        py::arg("v_rope"),
        py::arg("segment"),
        py::arg("layer_idx"),
        py::arg("bucket"));

#ifdef ENABLE_ATEN
    m.def(
        "warmup_piecewise_inductor_pre_attn",
        [](const infinicore::Tensor &positions,
           infinicore::Tensor &positions_padded,
           const infinicore::Tensor &hidden_states,
           const infinicore::Tensor &residual,
           size_t layer_idx,
           size_t bucket,
           size_t valid_len) {
            infinicore::op::inductor_segment_impl::warmup_pre_attn(
                positions,
                positions_padded,
                hidden_states,
                residual,
                layer_idx,
                bucket,
                valid_len);
        },
        py::arg("positions"),
        py::arg("positions_padded"),
        py::arg("hidden_states"),
        py::arg("residual"),
        py::arg("layer_idx"),
        py::arg("bucket"),
        py::arg("valid_len"));
#endif
}

} // namespace infinicore::compiled_subgraphs
