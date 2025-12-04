#ifndef __AMINMAX_CPU_H__
#define __AMINMAX_CPU_H__

#include "../../../../utils.h"
#include "../../../operator.h"
#include <vector>

namespace op::aminmax::cpu {

struct AminmaxInfo {
    std::vector<size_t> in_shape;
    std::vector<ptrdiff_t> in_strides;
    std::vector<size_t> out_shape;
    std::vector<ptrdiff_t> out_strides;
    std::vector<int64_t> reduce_dims;
    bool keepdim;
    bool is_global;
    infiniDtype_t dtype;

    static utils::Result<AminmaxInfo> create(
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t output_desc,
        int64_t dim,
        int32_t keepdim,
        int32_t has_dim);
};

class Descriptor : public InfiniopDescriptor {
public:
    AminmaxInfo _info;

    Descriptor(AminmaxInfo info, infiniDevice_t device, int device_id)
        : InfiniopDescriptor{device, device_id}, _info(std::move(info)) {}

    ~Descriptor() = default;

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t min_output_desc,
        infiniopTensorDescriptor_t max_output_desc,
        infiniopTensorDescriptor_t input_desc,
        int64_t dim,
        int32_t keepdim,
        int32_t has_dim);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *min_output,
        void *max_output,
        const void *input,
        void *stream) const;
};

} // namespace op::aminmax::cpu

#endif // __AMINMAX_CPU_H__
