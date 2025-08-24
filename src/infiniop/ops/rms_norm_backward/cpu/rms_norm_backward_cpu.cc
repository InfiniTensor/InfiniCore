#include "rms_norm_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::rms_norm_backward::cpu {

template <typename Tdata>
infiniStatus_t calculate_rms_norm_backward(
    const RMSNormBackwardInfo &info,
	Tdata * grad_x,
	Tdata * grad_w,
	const Tdata * grad_y,
	const Tdata * x,
	const Tdata * w,
    Tdata * workspace
) {
//  -------------------------------- start: perform operator on CPU --------------------------------

    Tdata * batch_grad_w = workspace;

    std::vector<ptrdiff_t> contiguous_stride(info.ndim - 1);
    size_t norm_dim = info.ndim - 1, norm_size = info.normalized_size();
	ptrdiff_t last_dim = 1;
    ptrdiff_t last_stride = 1;
    for(size_t d = 0; d < info.ndim - 1; d ++)
    {
        contiguous_stride[d] = last_dim * last_stride;  
        last_dim = info.x_shape[d];
        last_stride = contiguous_stride[d];
    }
    size_t batch_size = last_dim * last_stride;
    // memset
    for (size_t c = 0; c < norm_size; c++) {
        *(grad_w + c * info.grad_w_strides[0]) = utils::cast<Tdata>(0.);
    }


    #pragma omp parallel for
    for(size_t n = 0; n < batch_size; n ++) {
        auto grad_x_ptr = grad_x;
        auto grad_y_ptr = grad_y;
        auto grad_w_ptr = batch_grad_w + n;
        auto x_ptr = x;
        
        size_t rem = n;
        for (int d = info.ndim - 2; d >= 0; d --)
        {
            size_t dim_index = rem / contiguous_stride[d];
            rem = rem % contiguous_stride[d];
            grad_x_ptr += dim_index * info.grad_x_strides[d];
            grad_y_ptr += dim_index * info.grad_y_strides[d];
            x_ptr += dim_index * info.x_strides[d];
        }

        float ss = op::common_cpu::reduce_op::sumSquared(
            x_ptr, norm_size, info.x_strides[norm_dim]
        );
        float rms = 1. / std::sqrt(ss / norm_size);
        float sum_grad_y_times_y = 0.;
        for (size_t c = 0; c < norm_size; c++) {
            
            float grad_y_times_normed_x = utils::cast<float>(*(grad_y_ptr + c * info.grad_y_strides[norm_dim])) * \
                                        utils::cast<float>(*(x_ptr + c * info.x_strides[norm_dim])) * rms;
            // *(grad_w_ptr) = utils::cast<Tdata>(
            //     utils::cast<float>(*grad_w_ptr) + grad_y_times_normed_x
            // );
            *(grad_w_ptr + c * info.batch_size) = utils::cast<Tdata>(grad_y_times_normed_x);

            sum_grad_y_times_y += grad_y_times_normed_x * utils::cast<float>(*(w + c * info.w_strides[0]));
        }
        for (size_t c = 0; c < norm_size; c++) {
            *(grad_x_ptr + c * info.grad_x_strides[norm_dim]) = utils::cast<Tdata>(
                rms * (
                    utils::cast<float>(*(w + c * info.w_strides[0])) * utils::cast<float>(*(grad_y_ptr + c * info.grad_y_strides[norm_dim])) - \
                    rms * sum_grad_y_times_y * utils::cast<float>(*(x_ptr + c * info.x_strides[norm_dim])) / norm_size
                )
            );
        }        

    }
    #pragma omp barrier

    #pragma omp parallel for
    for (size_t c = 0; c < norm_size; c++) {
        *(grad_w + c * info.grad_w_strides[0]) = utils::cast<Tdata>(op::common_cpu::reduce_op::sum(
            batch_grad_w + c * info.batch_size, info.batch_size, 1
        ));
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
	infiniopTensorDescriptor_t grad_y_desc,
	infiniopTensorDescriptor_t x_desc,
	infiniopTensorDescriptor_t w_desc
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = grad_x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);
    


    auto result = RMSNormBackwardInfo::createRMSNormBackwardInfo(
		grad_x_desc,
		grad_w_desc,
		grad_y_desc,
		x_desc,
		w_desc
    );
    CHECK_RESULT(result);
    const RMSNormBackwardInfo &info = result.take();
    size_t WorkSpaceSize = info.total_size * infiniSizeOf(dtype);
//  ---------------------- end: check data type and calculate workspace size -----------------------
    
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id
    );    

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_RMS_NORM_BACKWARD(TDATA) \
    CHECK_STATUS(calculate_rms_norm_backward<TDATA>(_info, \
(TDATA *)grad_x, (TDATA *)grad_w, (const TDATA *)grad_y, (const TDATA *)x, (const TDATA *)w, (TDATA *)workspace))

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
	void * grad_x,
	void * grad_w,
	const void * grad_y,
	const void * x,
	const void * w,
    void *stream
) const {
    if (_info.dtype == INFINI_DTYPE_F16) {
        CALCULATE_RMS_NORM_BACKWARD(fp16_t);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CALCULATE_RMS_NORM_BACKWARD(bf16_t);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CALCULATE_RMS_NORM_BACKWARD(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
}
