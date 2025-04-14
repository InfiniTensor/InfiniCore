#include "swiglu_kunlun.h"
#include "../../../devices/kunlun/kunlun_handle.h"
#include "../../../devices/kunlun/kunlun_type.h"
#include <memory>
#include <stdint.h>

void swiglu_f32(kunlun_size_t c_data_size,
                kunlun_size_t ndim,
                bool contiguous,
                bool broadcasted, const kunlun_size_t *c_shape, const kunlun_ptrdiff_t *c_strides, const kunlun_size_t *a_shape, const kunlun_ptrdiff_t *a_strides,
                const kunlun_size_t *b_shape, const kunlun_ptrdiff_t *b_strides, float *c, const float *a, const float *b, XPUStream stream);

namespace op::swiglu::kunlun {

struct Descriptor::Opaque {
    std::shared_ptr<device::kunlun::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t up_desc,
    infiniopTensorDescriptor_t gate_desc) {

    auto handle = reinterpret_cast<device::kunlun::Handle *>(handle_);
    auto dtype = out_desc->dtype();
    const auto &out_shape = out_desc->shape();
    const auto &up_shape = up_desc->shape();
    const auto &gate_shape = gate_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(out_shape, up_shape, gate_shape);

    op::binary::BinaryInfo info;
    CHECK_STATUS(op::binary::createBinaryInfo(info, out_desc, up_desc, gate_desc));

    // Create descriptor
    *desc_ptr = new Descriptor(
        dtype,
        std::move(info),
        new Descriptor::Opaque{static_cast<device::kunlun::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *c,
    const void *a,
    const void *b,
    void *stream) const {
    kunlun_size_t c_data_size = _info.c_data_size;
    kunlun_size_t ndim = _info.ndim;
    bool contiguous = _info.contiguous;
    bool broadcasted = _info.broadcasted;

    char *tmp = (char *)malloc(3 * ndim * (sizeof(kunlun_size_t) + sizeof(kunlun_ptrdiff_t))); // 昆仑芯涉及的int64,uint64等数据类型必须全部用kunlun_ptrdiff_t取代
    char *tmp_stride = tmp + 3 * ndim * sizeof(kunlun_size_t);
    kunlun_size_t *c_shape = (kunlun_size_t *)tmp;
    kunlun_size_t *a_shape = c_shape + ndim;
    kunlun_size_t *b_shape = a_shape + ndim;

    kunlun_ptrdiff_t *c_strides = (kunlun_ptrdiff_t *)tmp_stride;
    kunlun_ptrdiff_t *a_strides = c_strides + ndim;
    kunlun_ptrdiff_t *b_strides = a_strides + ndim;
    for (kunlun_size_t i = 0; i < ndim; i++) {
        c_strides[i] = _info.c_strides.data()[i];
        a_strides[i] = _info.a_strides.data()[i];
        b_strides[i] = _info.b_strides.data()[i];
        c_shape[i] = _info.c_shape.data()[i];
        a_shape[i] = _info.a_shape.data()[i];
        b_shape[i] = _info.b_shape.data()[i];
    }

    switch (_dtype) {
    case INFINI_DTYPE_F32:

        swiglu_f32(c_data_size,
                   ndim,
                   contiguous,
                   broadcasted, c_shape, c_strides, a_shape, a_strides,
                   b_shape, b_strides, (float *)c, (float *)a, (float *)b, reinterpret_cast<kunlunStream_t>(stream));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    free(tmp);

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu::kunlun
