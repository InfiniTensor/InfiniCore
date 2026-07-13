#pragma once

#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include "infinicore/ops/inductor_segment.hpp"

namespace py = pybind11;

namespace infinicore::compiled_subgraphs {

namespace {

std::mutex g_pre_attn_external_weights_mutex;
std::unordered_map<size_t, infinicore::op::inductor_segment_impl::PreAttnExternalWeightTensors>
    g_pre_attn_external_weights;

void ensure_py_pre_attn_weight_resolver() {
    static std::once_flag once;
    std::call_once(once, []() {
        infinicore::op::inductor_segment_impl::set_pre_attn_weight_resolver(
            [](size_t layer_idx) -> infinicore::op::inductor_segment_impl::PreAttnExternalWeightTensors {
                std::lock_guard<std::mutex> lock(g_pre_attn_external_weights_mutex);
                auto it = g_pre_attn_external_weights.find(layer_idx);
                if (it == g_pre_attn_external_weights.end()) {
                    throw std::runtime_error(
                        "InductorSegment: pre_attn external weights not registered for layer "
                        + std::to_string(layer_idx));
                }
                return it->second;
            });
    });
}

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
        "register_pre_attn_external_weights",
        [](size_t layer_idx,
           const infinicore::Tensor &ln_weight,
           const infinicore::Tensor &q_weight,
           const infinicore::Tensor &k_weight,
           const infinicore::Tensor &v_weight,
           const infinicore::Tensor &q_norm_weight,
           const infinicore::Tensor &k_norm_weight) {
            ensure_py_pre_attn_weight_resolver();
            std::lock_guard<std::mutex> lock(g_pre_attn_external_weights_mutex);
            g_pre_attn_external_weights[layer_idx] =
                infinicore::op::inductor_segment_impl::PreAttnExternalWeightTensors{
                    ln_weight,
                    q_weight,
                    k_weight,
                    v_weight,
                    q_norm_weight,
                    k_norm_weight,
                };
        },
        py::arg("layer_idx"),
        py::arg("ln_weight"),
        py::arg("q_weight"),
        py::arg("k_weight"),
        py::arg("v_weight"),
        py::arg("q_norm_weight"),
        py::arg("k_norm_weight"));

    m.def("clear_pre_attn_external_weights", []() {
        {
            std::lock_guard<std::mutex> lock(g_pre_attn_external_weights_mutex);
            g_pre_attn_external_weights.clear();
        }
        infinicore::op::inductor_segment_impl::clear_pre_attn_weight_resolver();
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
