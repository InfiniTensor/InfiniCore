#include "tril_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::tril::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
	infiniopTensorDescriptor_t output_desc,
	infiniopTensorDescriptor_t input_desc,
	int diagonal
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = input_desc->dtype();
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------

    auto result = TrilInfo::createTrilInfo(
		output_desc,
		input_desc,
		diagonal
    );
    CHECK_RESULT(result);
    const TrilInfo &info = result.take();
    
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
	void * output,
	const void * input,
    void *stream
) const {

    #pragma omp parallel for
    for (int row = 0; row < _info.row_size; row ++)
    {
        auto output_ptr = reinterpret_cast<char *>(output) + row * _info.column_size * _info.elem_size;
        auto input_ptr = reinterpret_cast<const char *>(input) + row * _info.column_size * _info.elem_size;
        if (0 > row + _info.diagonal)
            memset(
                output_ptr,
                0,
                _info.elem_size * _info.column_size
            );
        else if (_info.column_size - 1 <= row + _info.diagonal)
            memcpy(
                output_ptr,
                input_ptr, 
                _info.elem_size * _info.column_size
            );
        else {
            memset(
                output_ptr + _info.elem_size * (row + _info.diagonal + 1),
                0,
                _info.elem_size * (_info.column_size - row - _info.diagonal - 1)
            );
            memcpy(
                output_ptr,
                input_ptr,
                _info.elem_size * (row + _info.diagonal + 1)
            );   
        }
    }    
    return INFINI_STATUS_SUCCESS;
}
}
