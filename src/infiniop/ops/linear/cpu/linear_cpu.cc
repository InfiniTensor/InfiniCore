#include "linear_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::linear::cpu {

template <typename Tdata>
infiniStatus_t calculate_linear(
    const LinearInfo &info,
    Tdata * y,
    const Tdata * x,
    const Tdata * w,
    const Tdata * b
) {
//  -------------------------------- start: perform operator on CPU --------------------------------
    #pragma omp parallel for
    for(size_t j = 0; j < info.out_features; j ++)
    {
        auto w_ptr = w + j * info.w_stride_out;
        float y_sum = info.bias ? (utils::cast<float>(*(b + j * info.b_stride))) : 0.;
        for(size_t i = 0; i < info.in_features; i ++)
        {
            y_sum += utils::cast<float>(*(x + i * info.x_stride)) * utils::cast<float>(*(w_ptr + i * info.w_stride_in));
        }
        *(y + j * info.y_stride) = utils::cast<Tdata>(y_sum);
    }
//  --------------------------------- end: perform operator on CPU ---------------------------------
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = y_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------

    auto result = LinearInfo::createLinearInfo(
        y_desc,
        x_desc,
        w_desc,
        b_desc
    );
    CHECK_RESULT(result);
    const LinearInfo &info = result.take();
    
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
    void * y,
    const void * x,
    const void * w,
    const void * b,
    void *stream
) const {
    #define CALCULATE_LINEAR(TDATA) \
        CHECK_STATUS(calculate_linear<TDATA>(_info, \
    (TDATA *)y, (const TDATA *)x, (const TDATA *)w, (const TDATA *)b))    
    if (_info.dtype == INFINI_DTYPE_F16) {
        CALCULATE_LINEAR(fp16_t);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CALCULATE_LINEAR(bf16_t);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CALCULATE_LINEAR(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    #undef CALCULATE_LINEAR

    return INFINI_STATUS_SUCCESS;
}
}
