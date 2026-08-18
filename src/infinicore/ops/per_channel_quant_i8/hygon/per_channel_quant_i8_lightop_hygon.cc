#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/per_channel_quant_i8.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/lightop_adaptor.hpp"

#include <ATen/ATen.h>
#include <c10/hip/HIPGuard.h>

#include <stdexcept>

namespace infinicore::op::per_channel_quant_i8_impl::lightop_hygon {

struct PlannedMeta {
    graph::GraphTensor x;
    graph::GraphTensor x_packed;
    graph::GraphTensor x_scale;
    at::Tensor smooth;
};

void *plan(const Tensor &x, Tensor x_packed, Tensor x_scale) {
    infinicore::adaptor::lightop::preload_w8a8_linear_ops();
    if (x->ndim() != 2 || x_packed->ndim() != 2 || x_scale->ndim() != 2) {
        throw std::runtime_error("Hygon per_channel_quant_i8 expects 2D tensors");
    }
    const auto m = x->shape()[0];
    const auto k = x->shape()[1];
    if (x_packed->shape() != x->shape() || x_scale->shape() != std::vector<Size>{m, 1}) {
        throw std::runtime_error("Hygon per_channel_quant_i8 shape mismatch");
    }
    if (x_packed->dtype() != DataType::I8 || x_scale->dtype() != DataType::F32) {
        throw std::runtime_error("Hygon per_channel_quant_i8 output dtype mismatch");
    }

    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(infinicore::adaptor::to_at_device(x->device()))
                       .requires_grad(false);
    auto smooth = at::ones({static_cast<int64_t>(k)}, options);

    return new PlannedMeta{
        graph::GraphTensor(x),
        graph::GraphTensor(x_packed),
        graph::GraphTensor(x_scale),
        smooth};
}

void run(void *planned_meta) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    Tensor x_work = p->x->is_contiguous() ? Tensor(p->x) : p->x->contiguous();

    const bool packed_need_copy_back = !p->x_packed->is_contiguous();
    const bool scale_need_copy_back = !p->x_scale->is_contiguous();
    Tensor packed_work = packed_need_copy_back ? p->x_packed->contiguous() : Tensor(p->x_packed);
    Tensor scale_work = scale_need_copy_back ? p->x_scale->contiguous() : Tensor(p->x_scale);

    auto x = infinicore::adaptor::to_aten_tensor(x_work);
    auto x_packed = infinicore::adaptor::to_aten_tensor(packed_work);
    auto x_scale = infinicore::adaptor::to_aten_tensor(scale_work);

    infinicore::adaptor::lightop::per_token_dynamic_quant_int8(
        x_packed, x, x_scale, p->smooth);

    if (packed_need_copy_back) {
        p->x_packed->copy_from(packed_work);
    }
    if (scale_need_copy_back) {
        p->x_scale->copy_from(scale_work);
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
    PerChannelQuantI8::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    PerChannelQuantI8::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    PerChannelQuantI8::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::per_channel_quant_i8_impl::lightop_hygon

#endif // ENABLE_HYGON_API && ENABLE_ATEN
