#include "linear_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::linear_backward::cpu {

template <typename Tdata>
infiniStatus_t calculate_linear_backward(
    const LinearBackwardInfo &info,
	Tdata * grad_x,
	Tdata * grad_w,
	Tdata * grad_b,
	const Tdata * grad_y,
	const Tdata * x,
	const Tdata * w
) {
//  -------------------------------- start: perform operator on CPU --------------------------------
    // #pragma omp parallel for
    for (size_t i = 0; i < info.in_features; i ++)
    {
        auto w_ptr = w + i * info.w_stride_in;
        auto grad_w_ptr = grad_w + i * info.grad_w_stride_in;
        float grad_x_sum = 0.;
        float x_value = utils::cast<float>(*(x + i * info.x_stride));
        for (size_t j = 0; j < info.out_features; j ++)
        {
            float grad_y_value = utils::cast<float>(*(grad_y + j * info.grad_y_stride));
            grad_x_sum += grad_y_value * utils::cast<float>(*(w_ptr + j * info.w_stride_out));
            (*(grad_w_ptr + j * info.grad_w_stride_out)) = utils::cast<Tdata>(x_value * grad_y_value);
            if (info.bias && i == 0)
                (*(grad_b + j * info.grad_b_stride)) = utils::cast<Tdata>(grad_y_value);
        }

        *(grad_x + i * info.grad_x_stride) = utils::cast<Tdata>(grad_x_sum);
    }
//  --------------------------------- end: perform operator on CPU ---------------------------------
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
	infiniopTensorDescriptor_t grad_x_desc,
	infiniopTensorDescriptor_t grad_w_desc,
	infiniopTensorDescriptor_t grad_b_desc,
	infiniopTensorDescriptor_t grad_y_desc,
	infiniopTensorDescriptor_t x_desc,
	infiniopTensorDescriptor_t w_desc
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = grad_y_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------

    auto result = LinearBackwardInfo::createLinearBackwardInfo(
		grad_x_desc,
		grad_w_desc,
		grad_b_desc,
		grad_y_desc,
		x_desc,
		w_desc
    );
    CHECK_RESULT(result);
    const LinearBackwardInfo &info = result.take();
    
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
	void * grad_x,
	void * grad_w,
	void * grad_b,
	const void * grad_y,
	const void * x,
	const void * w,
    void *stream
) const {

    #define CALCULATE_LINEAR_BACKWARD(TDATA) \
        CHECK_STATUS(calculate_linear_backward<TDATA>(_info, \
    (TDATA *)grad_x, (TDATA *)grad_w, (TDATA *)grad_b, (const TDATA *)grad_y, (const TDATA *)x, (const TDATA *)w))

    if (_info.dtype == INFINI_DTYPE_F16) {
        CALCULATE_LINEAR_BACKWARD(fp16_t);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CALCULATE_LINEAR_BACKWARD(bf16_t);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CALCULATE_LINEAR_BACKWARD(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    #undef CALCULATE_LINEAR_BACKWARD

    return INFINI_STATUS_SUCCESS;
}
}
