#include "index_copy_inplace_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../../rearrange/cpu/rearrange_cpu.h"
#include "../info.h"

namespace op::index_copy_inplace::cpu {

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


    auto result = IndexCopyInplaceInfo::createIndexCopyInplaceInfo(
        output_desc,
        input_desc,
        index_desc,
        dim
    );
    CHECK_RESULT(result);
    const IndexCopyInplaceInfo &info = result.take();
    size_t WorkSpaceSize = (info.total_input_size + info.total_output_size) * infiniSizeOf(dtype);
//  ---------------------- end: check data type and calculate workspace size -----------------------    
    InfiniopTensorDescriptor * rearrange_in_desc = new InfiniopTensorDescriptor(
        dtype, input_desc->ndim(), input_desc->shape().data(), info.meta_strides.data()
    );
    InfiniopTensorDescriptor * rearrange_out_desc = new InfiniopTensorDescriptor(
        dtype, input_desc->ndim(), output_desc->shape().data(), info.meta_strides.data()
    );        
    
    void * in_rearrange_descriptor = nullptr;
    void * out_rearrange_descriptor = nullptr;

    op::rearrange::cpu::Descriptor::create(
        handle_, reinterpret_cast<op::rearrange::cpu::Descriptor **>(&in_rearrange_descriptor),
        rearrange_in_desc, input_desc
    );
    op::rearrange::cpu::Descriptor::create(
        handle_, reinterpret_cast<op::rearrange::cpu::Descriptor **>(&out_rearrange_descriptor),
        output_desc, rearrange_out_desc
    );

    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id,
        in_rearrange_descriptor,
        out_rearrange_descriptor
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
    size_t size_of_dtype = infiniSizeOf(_info.dtype);
    auto index_ptr = reinterpret_cast<const int64_t *>(index);


    char* workspace_in = reinterpret_cast<char*>(workspace);
    char* workspace_out = workspace_in + size_of_dtype * _info.total_input_size;
    
    
    reinterpret_cast<op::rearrange::cpu::Descriptor *>(_rearrange_desc_in)->calculate(workspace_in, input, stream);
    memset(workspace_out, 0, _info.total_output_size * size_of_dtype);
    size_t copy_unit_size = _info.meta_strides[_info.dim] * size_of_dtype;
    #pragma omp parallel for
    for (size_t dst_index = 0; dst_index < _info.output_shape[_info.dim]; dst_index++) {
        size_t src_index = _info.index_shape[0] - 1;
        while (true)
        {
            if (*(index_ptr + src_index * _info.index_strides[0]) == int64_t(dst_index)) {
                std::memcpy(
                    workspace_out + size_of_dtype * dst_index * _info.meta_strides[_info.dim],
                    workspace_in + size_of_dtype * src_index * _info.meta_strides[_info.dim],
                    copy_unit_size
                );
                break;
            }
            else if (src_index == 0)
                break;
            src_index --;
        }
    }
    reinterpret_cast<op::rearrange::cpu::Descriptor *>(_rearrange_desc_out)->calculate(output, workspace_out, stream);

    return INFINI_STATUS_SUCCESS;
}
}
