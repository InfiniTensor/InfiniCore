#ifdef INFINIOP_SUM_DESCRIPTOR_H_
#define INFINIOP_SUM_DESCRIPTOR_H_
#include "../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/sum.h"
#include <vector>
#include <cstddef>

#define DESCRIPTOR(NAMESPACE) \
    namespace op::sum::NAMESPACE { \
    class Descriptor final : public InfiniopDescriptor { \
        struct Opaque; \
        Opaque *_opaque; \
        SumInfo _info; \
        size_t _workspace_size; \
        Descriptor( \
            Opaque *opaque, \
            SumInfo info, \
            size_t workspace_size, \
            infiniDevice_t device_type, \
            int device_id) \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque), \
              _info(info), \
              _workspace_size(workspace_size) {} \
        public: \
        ~Descriptor(); \

        size_t workspaceSize() const { return _workspace_size; } \

        static infiniStatus_t create( \
            infiniopHandle_t handle, \
            Descriptor **desc_ptr, \
            infiniopTensorDescriptor_t output_desc, \
            infiniopTensorDescriptor_t input_desc, \
            std::vector<int32_t> dim, \
            bool keepdim); \
        
        infiniStatus_t calculate(   \
            void *workspace, \
            size_t workspace_size, \
            void *output, \
            const void *input, \
            std::vector<int32_t> dim, \
            bool keepdim, \
            void *stream) const; \
    }; \
}

class SumInfo{
    private:
        SumInfo() = default;
    public:
        infiniDtype_t dtype;
        std::vector<std::size_t> in_shape;
        std::vector<std::size_t> out_shape;
        std::vector<ptrdiff_t> in_strides;
        std::vector<ptrdiff_t> out_strides;
        static utils::Result<SumInfo> create(
            infiniopTensorDescriptor_t output_desc,
            infiniopTensorDescriptor_t input_desc,
            std::vector<int32_t> dim, // todo 后续跑之前把int32_t都转成size_t
            bool keepdim){
                CHECK_OR_RETURN(output_desc != nullptr && input_desc != nullptr, INFINI_STATUS_NULL_POINTER); 
                CHECK_OR_RETURN(output_desc->dtype() == input_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
                in_shape = input_desc->shape();
                out_shape = output_desc->shape();
                in_strides = input_desc->strides();
                out_strides = output_desc->strides();
                dtype = input_desc->dtype();
                return utils::Result<SumInfo>(SumInfo{dtype, in_shape, out_shape, in_strides, out_strides});
            }
        size_t workspaceSize() const { return _workspace_size; }
};
#endif
