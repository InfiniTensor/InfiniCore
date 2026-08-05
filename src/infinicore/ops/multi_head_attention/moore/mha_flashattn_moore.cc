#if defined(ENABLE_MOORE_MATE_FLASH_ATTN)

#include "infinicore/ops/mha.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/csrc/utils/pybind.h>

namespace infinicore::op::mha_impl::flashattn_moore {

namespace py = pybind11;

namespace {
class LocalMUSAStreamGuard {
public:
    explicit LocalMUSAStreamGuard(const c10::musa::MUSAStream &s)
        : prev_(c10::musa::getCurrentMUSAStream(s.device_index())) {
        c10::musa::setCurrentMUSAStream(s);
    }
    ~LocalMUSAStreamGuard() {
        c10::musa::setCurrentMUSAStream(prev_);
    }
    LocalMUSAStreamGuard(const LocalMUSAStreamGuard &) = delete;
    LocalMUSAStreamGuard &operator=(const LocalMUSAStreamGuard &) = delete;

private:
    c10::musa::MUSAStream prev_;
};
} // namespace

struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    bool is_causal;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        is_causal};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    if (p->alibi_slopes.has_value()) {
        throw std::runtime_error(
            "[mha/moore] ALiBi is not supported by mate v0.1.3");
    }

    LocalMUSAStreamGuard guard(infinicore::adaptor::get_musa_stream());

    auto out = infinicore::adaptor::to_aten_tensor(p->out);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    try {
        py::gil_scoped_acquire gil;
        py::module_ wrapper = py::module_::import(
            "infinicore.ops.moore_mate_flash_attn");

        py::object result = wrapper.attr("moore_mate_flash_attn_dense")(
            py::cast(q),
            py::cast(k),
            py::cast(v),
            p->scale,
            p->is_causal);

        out.copy_(result.cast<at::Tensor>());
    } catch (const py::error_already_set &e) {
        throw std::runtime_error(
            std::string("[mha/moore] Python error: ") + e.what());
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::MOORE, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::MOORE, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::MOORE, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::flashattn_moore

#endif // ENABLE_MOORE_MATE_FLASH_ATTN
