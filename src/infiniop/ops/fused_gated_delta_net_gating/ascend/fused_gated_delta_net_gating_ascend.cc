#include "fused_gated_delta_net_gating_ascend.h"
#include "../../../devices/ascend/ascend_handle.h"

namespace op::fused_gated_delta_net_gating::ascend {
extern "C" infiniStatus_t fused_gated_delta_net_gating_kernel_launch(
    void *g, void *beta_output,
    const void *A_log, const void *a, const void *b, const void *dt_bias,
    infiniDtype_t input_dtype, infiniDtype_t parameter_dtype,
    size_t total, size_t seq_len, size_t hidden,
    ptrdiff_t g_s0, ptrdiff_t g_s1, ptrdiff_t g_s2,
    ptrdiff_t beta_s0, ptrdiff_t beta_s1, ptrdiff_t beta_s2,
    ptrdiff_t A_log_s0,
    ptrdiff_t a_s0, ptrdiff_t a_s1, ptrdiff_t a_s2,
    ptrdiff_t b_s0, ptrdiff_t b_s1, ptrdiff_t b_s2,
    ptrdiff_t dt_bias_s0, float beta, float threshold, void *stream);

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_output_desc,
    infiniopTensorDescriptor_t A_log_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t dt_bias_desc,
    float beta,
    float threshold) {

    auto result = FusedGatedDeltaNetGatingInfo::create(
        g_desc, beta_output_desc, A_log_desc, a_desc, b_desc, dt_bias_desc,
        beta, threshold, true);
    CHECK_RESULT(result);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{}, result.take(), 0,
        handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *g,
    void *beta_output,
    const void *A_log,
    const void *a,
    const void *b,
    const void *dt_bias,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    return fused_gated_delta_net_gating_kernel_launch(
        g, beta_output, A_log, a, b, dt_bias,
        _info.input_dtype, _info.parameter_dtype,
        _info.numel(), _info.seq_len, _info.hidden,
        _info.g_strides[0], _info.g_strides[1], _info.g_strides[2],
        _info.beta_output_strides[0], _info.beta_output_strides[1],
        _info.beta_output_strides[2], _info.A_log_strides[0],
        _info.a_strides[0], _info.a_strides[1], _info.a_strides[2],
        _info.b_strides[0], _info.b_strides[1], _info.b_strides[2],
        _info.dt_bias_strides[0], _info.beta, _info.threshold, stream);
}

} // namespace op::fused_gated_delta_net_gating::ascend
