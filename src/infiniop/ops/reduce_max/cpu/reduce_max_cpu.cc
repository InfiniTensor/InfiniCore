#include "reduce_max_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::reduce_max::cpu {

template <typename Tdata>
infiniStatus_t calculate_reduce_max(
    const ReduceMaxInfo &info,
	Tdata * output,
	const Tdata * input
) {
//  -------------------------------- start: perform operator on CPU --------------------------------
    std::vector<ptrdiff_t> natural_stride(info.ndim, 0);
	ptrdiff_t last_dim = 1;
    ptrdiff_t last_stride = 1;
    for(size_t d = 0; d < info.ndim; d ++)
    {
        if (d == info.dim)
            continue;
        else {
            natural_stride[d] = last_dim * last_stride;  
            last_dim = info.input_shape[d];
            last_stride = natural_stride[d];
        }
    }
    size_t batch_size = last_dim * last_stride;

    #pragma omp parallel for
    for(size_t n = 0; n < batch_size; n ++)
    {
        auto output_ptr = output;
        auto input_ptr = input;
        size_t rem = n;
        for(int d = info.ndim - 1; d >= 0; d --)
        {
            if(d == int(info.dim))
                continue;
            size_t dim_index = rem / natural_stride[d];
            rem = rem % natural_stride[d];
            output_ptr += dim_index * info.output_strides[d];
            input_ptr += dim_index * info.input_strides[d];
        }
        *output_ptr = utils::cast<Tdata>(op::common_cpu::reduce_op::max(
            input_ptr,
            info.input_shape[info.dim],
            info.input_strides[info.dim]
        ));
    }
//  --------------------------------- end: perform operator on CPU ---------------------------------
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
	infiniopTensorDescriptor_t output_desc,
	infiniopTensorDescriptor_t input_desc,
	size_t dim
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------

    auto result = ReduceMaxInfo::createReduceMaxInfo(
		output_desc,
		input_desc,
		dim
    );
    CHECK_RESULT(result);
    const ReduceMaxInfo &info = result.take();
    
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id
    );    

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_REDUCE_MAX(TDATA) \
    CHECK_STATUS(calculate_reduce_max<TDATA>(_info, \
(TDATA *)output, (const TDATA *)input))

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
	void * output,
	const void * input,
    void *stream
) const {
    if (_info.dtype == INFINI_DTYPE_F16) {
        CALCULATE_REDUCE_MAX(fp16_t);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CALCULATE_REDUCE_MAX(bf16_t);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CALCULATE_REDUCE_MAX(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
}
