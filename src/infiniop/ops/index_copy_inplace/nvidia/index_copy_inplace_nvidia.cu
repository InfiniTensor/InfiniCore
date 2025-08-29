#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "index_copy_inplace_nvidia.cuh"
#include "../../rearrange/nvidia/rearrange_nvidia.cuh"
#include "../info.h"

namespace op::index_copy_inplace::nvidia {

infiniStatus_t calculate_index_copy_inplace(
    char * output,
    const char * input,
    const int64_t * index,
    size_t copy_unit_size,
    size_t output_len,
    size_t index_len,
    ptrdiff_t index_stride,
    cudaStream_t stream
) {
    int64_t* dst_index = new int64_t;
    size_t sizeof_int64_t = sizeof(int64_t);
    for (size_t src_index = 0; src_index < index_len; src_index ++) {
        CHECK_CUDA(cudaMemcpyAsync(
            dst_index,
            index + src_index * index_stride,
            sizeof_int64_t,
            cudaMemcpyDeviceToHost,
            stream
        ));
        cudaStreamSynchronize(stream);
        CHECK_CUDA(cudaMemcpyAsync(
            output + (size_t)(*dst_index) * copy_unit_size,
            input + src_index * copy_unit_size,
            copy_unit_size,
            cudaMemcpyDeviceToDevice,
            stream
        ));
        cudaStreamSynchronize(stream);
    }
    delete dst_index;
    return INFINI_STATUS_SUCCESS;
}

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete reinterpret_cast<op::rearrange::nvidia::Descriptor *>(_rearrange_desc_in);
    delete reinterpret_cast<op::rearrange::nvidia::Descriptor *>(_rearrange_desc_out);    
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t index_desc,
    size_t dim
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = output_desc->dtype();
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = IndexCopyInplaceInfo::createIndexCopyInplaceInfo(
        output_desc,
        input_desc,
        index_desc,
        dim
    );
    CHECK_RESULT(result);
    const IndexCopyInplaceInfo &info = result.take();
    size_t WorkSpaceSize = (info.total_input_size + info.total_output_size) * infiniSizeOf(dtype);

    InfiniopTensorDescriptor * rearrange_in_desc = new InfiniopTensorDescriptor(
        dtype, input_desc->ndim(), input_desc->shape().data(), info.meta_strides.data()
    );
    InfiniopTensorDescriptor * rearrange_out_desc = new InfiniopTensorDescriptor(
        dtype, input_desc->ndim(), output_desc->shape().data(), info.meta_strides.data()
    );        
    
    void * in_rearrange_descriptor = nullptr;
    void * out_rearrange_descriptor = nullptr;

    op::rearrange::nvidia::Descriptor::create(
        handle_, reinterpret_cast<op::rearrange::nvidia::Descriptor **>(&in_rearrange_descriptor),
        rearrange_in_desc, input_desc
    );
    op::rearrange::nvidia::Descriptor::create(
        handle_, reinterpret_cast<op::rearrange::nvidia::Descriptor **>(&out_rearrange_descriptor),
        output_desc, rearrange_out_desc
    );    

    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        new Opaque{handle->internal()},
        handle->device, handle->device_id,
        in_rearrange_descriptor,
        out_rearrange_descriptor        
    );    
    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void * workspace,
    size_t workspace_size,
    void * output,
    const void * input,
    const void * index,
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    size_t elem_size = infiniSizeOf(_info.dtype);
    char* workspace_in = reinterpret_cast<char*>(workspace);
    char* workspace_out = workspace_in + elem_size * _info.total_input_size;
    CHECK_STATUS(reinterpret_cast<op::rearrange::nvidia::Descriptor *>(_rearrange_desc_in)->calculate(workspace_in, input, stream));
    cudaMemsetAsync(workspace_out, 0, _info.total_output_size * elem_size, stream);
    cudaDeviceSynchronize();
    CHECK_STATUS(calculate_index_copy_inplace(
        reinterpret_cast<char*>(workspace_out),
        reinterpret_cast<char*>(workspace_in),
        reinterpret_cast<const int64_t*>(index),
        elem_size * _info.meta_strides[_info.dim],
        _info.output_shape[_info.dim],
        _info.index_shape[0],
        _info.index_strides[0],
        stream
    ));
    cudaDeviceSynchronize();

    CHECK_STATUS(reinterpret_cast<op::rearrange::nvidia::Descriptor *>(_rearrange_desc_out)->calculate(output, workspace_out, stream));
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::index_copy_inplace::nvidia
