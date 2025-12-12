#include "../../../devices/cpu/common_cpu.h"
#include "argwhere_cpu.h"
#include "infinicore.h"
#include<array>
#include <cstddef>
#include <cstdint>
namespace op::argwhere::cpu {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc
    ) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info = ArgwhereInfo::create(x_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateArgWhere(
    const ArgwhereInfo &info,
    void *workspace,
    size_t workspace_size,
    void **y,
    size_t *count,
    const void *x) {

    const Tdata *x_data = reinterpret_cast<const Tdata *>(x);
    // int64_t *y_data = reinterpret_cast<int64_t *>(y);
    std::vector<size_t> positions;
    // #pragma omp parallel for
    for(size_t i = 0; i < info.num_elements; i++){
        size_t pos=0,tem=i;
        std::vector<size_t> position(info.strides.size());
        for(int j = info.strides.size()-1; j >=0; j--){
            position[j] = tem % info.shapes[j];
            tem /= info.shapes[j];
            pos += position[j] * info.strides[j];
        }
        if(fabs(x_data[pos]-0.0f) > 1e-5){
            for(auto p:position){
                positions.push_back(p);
            }
        }
    }

    *y=new int64_t[positions.size()];
    memcpy(*y, positions.data(), positions.size() * sizeof(int64_t));
    *count=positions.size()/info.strides.size();
    return INFINI_STATUS_SUCCESS;

}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size, 
    void **y,
    size_t *count,
    const void *x,
    void *stream) const{
    switch(_info.dtype){
        case INFINI_DTYPE_F32:
            return calculateArgWhere<float>(_info, workspace, workspace_size, y, count, x);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
    
}