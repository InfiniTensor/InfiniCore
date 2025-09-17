#include "equal_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::equal::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = c_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_BOOL);
    CHECK_OR_RETURN(b_desc->dtype() == a_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------

    auto result = EqualInfo::createEqualInfo(
        c_desc,
        a_desc,
        b_desc
    );
    CHECK_RESULT(result);
    const EqualInfo &info = result.take();
    
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id
    );    

    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void * c,
    const void * a,
    const void * b,
    void *stream
) const {
    std::vector<ptrdiff_t> contiguous_strides(_info.ndim);
	ptrdiff_t last_dim = 1;
    ptrdiff_t last_stride = 1;
    for(size_t d = 0; d < _info.ndim; d ++)
    {
        contiguous_strides[d] = last_dim * last_stride;  
        last_dim = _info.a_shape[d];
        last_stride = contiguous_strides[d];
    }
    size_t total_size = last_dim * last_stride;
    size_t elem_size = infiniSizeOf(_info.dtype);
    auto c_ptr = reinterpret_cast<bool*>(c);
    *c_ptr = true;
    #pragma omp parallel for
    for(size_t i = 0; i < total_size; i ++) {
        auto a_ptr = reinterpret_cast<const char*>(a);
        auto b_ptr = reinterpret_cast<const char*>(b);
        size_t rem = i;
        for(int d = _info.ndim - 1; d >= 0; d --) {
            size_t dim_index = rem / contiguous_strides[d];
            rem = rem % contiguous_strides[d];
            a_ptr += dim_index * _info.a_strides[d];
            b_ptr += dim_index * _info.b_strides[d];
        }
        if (memcmp(a_ptr, b_ptr, elem_size) != 0) {
            *c_ptr = false;
        }
    }
    return INFINI_STATUS_SUCCESS;
}
}
