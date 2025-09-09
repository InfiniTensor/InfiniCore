#include "gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::gather::cpu {

infiniStatus_t calculate_gather(
    const GatherInfo &info,
    char * output,
    const char * input,
    const int64_t * index
) {
//  -------------------------------- start: perform operator on CPU --------------------------------
    std::vector<ptrdiff_t> contiguous_strides(info.ndim);
	ptrdiff_t last_dim = 1;
    ptrdiff_t last_stride = 1;
    for(size_t d = 0; d < info.ndim; d ++)
    {
        contiguous_strides[d] = last_dim * last_stride;  
        last_dim = info.output_shape[d];
        last_stride = contiguous_strides[d];
    }
    size_t total_size = last_dim * last_stride;
    
    int gather_dim = info.dim;
    size_t element_size = infiniSizeOf(info.dtype);

    #pragma omp parallel for
    for(size_t i = 0; i < total_size; i++)
    {
        auto output_ptr = output;
        auto input_ptr = input;
        auto index_ptr = index;
        size_t rem = i;
        for (int d = info.ndim - 1; d >= 0; d--) {
            size_t dim_index = rem / contiguous_strides[d];
            rem = rem % contiguous_strides[d];
            output_ptr += dim_index * element_size * info.output_strides[d];
            index_ptr += dim_index * info.index_strides[d];
            if(d != gather_dim)
                input_ptr += dim_index * element_size * info.input_strides[d];
        }
        int64_t gather_number = *index_ptr;
        input_ptr += gather_number * element_size * info.input_strides[gather_dim];
        // *output_ptr = *input_ptr;
        memcpy(
            output_ptr,
            input_ptr,
            element_size
        );
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
    infiniopTensorDescriptor_t index_desc,
    size_t dim
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = output_desc->dtype();
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------

    auto result = GatherInfo::createGatherInfo(
        output_desc,
        input_desc,
        index_desc,
        dim
    );
    CHECK_RESULT(result);
    const GatherInfo &info = result.take();
    
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
    const void * index,
    void *stream
) const {

    return calculate_gather(_info, (char *)output, (const char *)input, (const int64_t *)index);
}
}

