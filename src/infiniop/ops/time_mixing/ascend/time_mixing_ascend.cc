#include "time_mixing_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

namespace op::time_mixing::ascend {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t y_desc,
                                  infiniopTensorDescriptor_t r_desc, infiniopTensorDescriptor_t w_desc,
                                  infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t v_desc,
                                  infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc) {
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    auto result = TimeMixingInfo::create(y_desc, r_desc, w_desc, k_desc, v_desc, a_desc, b_desc);
    CHECK_RESULT(result);
    TimeMixingInfo info = result.take();
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(std::move(info), workspace_size, handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

extern "C" infiniStatus_t
time_mixing_kernel_launch(void *y, void *r, void *w, void *k, void *v, void *a, void *b,
                          int B, int T, int C, int H, int N,
                          infiniDtype_t dt, void *stream);

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *y, const void *r, const void *w,
                                     const void *k, const void *v, const void *a, const void *b, void *stream) const {
    int B = _info.B;
    int T = _info.T;
    int C = _info.C;
    int H = _info.H;
    int N = _info.N;
    auto dt = _info.dtype;
    auto status = time_mixing_kernel_launch(y, (void *)r, (void *)w, (void *)k, (void *)v, (void *)a, (void *)b, B, T, C, H, N, dt, stream);
    return status;
}

} // namespace op::time_mixing::ascend