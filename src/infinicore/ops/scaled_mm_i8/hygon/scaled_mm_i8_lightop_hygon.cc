#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/scaled_mm_i8.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/lightop_adaptor.hpp"

#include <ATen/ATen.h>
#include <c10/hip/HIPGuard.h>

#include <stdexcept>

namespace infinicore::op::scaled_mm_i8_impl::lightop_hygon {

struct PlannedMeta {
    graph::GraphTensor c;
    graph::GraphTensor a_p;
    graph::GraphTensor a_s;
    graph::GraphTensor b_p;
    graph::GraphTensor b_s;
    std::optional<graph::GraphTensor> bias;
};

void *plan(Tensor c,
           const Tensor &a_p,
           const Tensor &a_s,
           const Tensor &b_p,
           const Tensor &b_s,
           std::optional<Tensor> bias) {
    infinicore::adaptor::lightop::preload_w8a8_linear_ops();
    if (c->ndim() != 2 || a_p->ndim() != 2 || b_p->ndim() != 2) {
        throw std::runtime_error("Hygon scaled_mm_i8 expects 2D tensors");
    }
    if (a_p->dtype() != DataType::I8 || b_p->dtype() != DataType::I8 ||
        a_s->dtype() != DataType::F32 || b_s->dtype() != DataType::F32) {
        throw std::runtime_error("Hygon scaled_mm_i8 expects int8 inputs and float32 scales");
    }
    if (a_p->shape()[0] != c->shape()[0] || b_p->shape()[1] != c->shape()[1] ||
        a_p->shape()[1] != b_p->shape()[0]) {
        throw std::runtime_error("Hygon scaled_mm_i8 matrix shape mismatch");
    }

    return new PlannedMeta{
        graph::GraphTensor(c),
        graph::GraphTensor(a_p),
        graph::GraphTensor(a_s),
        graph::GraphTensor(b_p),
        graph::GraphTensor(b_s),
        bias ? std::optional<graph::GraphTensor>(graph::GraphTensor(*bias)) : std::nullopt};
}

void run(void *planned_meta) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    const bool c_need_copy_back = !p->c->is_contiguous();
    Tensor c_work = c_need_copy_back ? p->c->contiguous() : Tensor(p->c);
    Tensor a_work = p->a_p->is_contiguous() ? Tensor(p->a_p) : p->a_p->contiguous();
    Tensor a_scale_work = p->a_s->is_contiguous() ? Tensor(p->a_s) : p->a_s->contiguous();
    Tensor b_scale_work = p->b_s->is_contiguous() ? Tensor(p->b_s) : p->b_s->contiguous();

    Tensor b_work = Tensor(p->b_p);

    auto c = infinicore::adaptor::to_aten_tensor(c_work);
    auto a = infinicore::adaptor::to_aten_tensor(a_work);
    auto a_scale = infinicore::adaptor::to_aten_tensor(a_scale_work);
    auto b_scale = infinicore::adaptor::to_aten_tensor(b_scale_work);
    std::optional<at::Tensor> bias = std::nullopt;
    if (p->bias.has_value()) {
        bias = infinicore::adaptor::to_aten_tensor(Tensor(p->bias.value()));
    }

    Tensor b_nk_work = b_work->permute({1, 0});
    if (!b_nk_work->is_contiguous()) {
        b_nk_work = b_nk_work->contiguous();
    }
    auto b_nk = infinicore::adaptor::to_aten_tensor(b_nk_work);
    infinicore::adaptor::lightop::blaslt_w8a8_gemm(
        c, a, b_nk, a_scale, b_scale, bias);

    if (c_need_copy_back) {
        p->c->copy_from(c_work);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    if (!infinicore::adaptor::lightop::available()) {
        return false;
    }
    I8Gemm::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    I8Gemm::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    I8Gemm::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::scaled_mm_i8_impl::lightop_hygon

#endif // ENABLE_HYGON_API && ENABLE_ATEN
