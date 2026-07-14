#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/concat_mla_q.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_concat_mla_q(py::module &m) {
    m.def("concat_mla_q_",
          &op::concat_mla_q_,
          py::arg("ql_nope"),
          py::arg("q_pe"),
          py::arg("q_out"),
          R"doc(In-place MLA query concat: q_out[..., :nope] = ql_nope; q_out[..., nope:] = q_pe.)doc");

    m.def("concat_mla_q",
          &op::concat_mla_q,
          py::arg("ql_nope"),
          py::arg("q_pe"),
          R"doc(MLA query concat via vllm_iluvatar on Iluvatar.)doc");
}

} // namespace infinicore::ops
