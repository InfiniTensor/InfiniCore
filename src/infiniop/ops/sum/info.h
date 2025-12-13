#ifndef __SUM_INFO_H__
#define __SUM_INFO_H__
#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>
#include <cstddef>
#include <algorithm>



// 在最外围计算reduce_tensor output_tensor的shape
// 假设reduce dim为 dim1 dim2 dim3... 其他dim记为 other1, other2, other3...  均为有序
// reduce_tesnor = input_tensor.permute(other1, other2, other3...dim1, dim2, dim3...)
// 计算有多少个值需要 reduce     reduce_num = shape[dim1] * shape[dim2] * shape[dim3] * ...
// 然后这个时候reduce_tensor的shape strides都变过来了
// for auto output_index  : output_size (整数)
// 计算output_index 对应的 output_offset 其实就是output_index
// 计算reduce_num个input_offset 相加 得到tempSum
// for size_t i : output_size
// convert(i * reduce_num) .... convert((i+1) * reduce_num - 1)
// indexToOffset 来进行计算 就行


// for(size_t i = 0; i < input_ndim; i++){
                //     if(std::find(dim, dim + dim_size, i) == dim + dim_size){
                //         permute_order.push_back(i);
                //     }
                // }
                // for(size_t i = 0; i < dim_size; i++){
                //     permute_order.push_back(dim[i]);
                // }

// 在info这里做permute就行了！
namespace op::sum{
    class SumInfo{
    SumInfo() = default;
    public:
        infiniDtype_t dtype;
        std::vector<size_t> permuted_input_shape; // need to permute shape for reduce tensor (other1, other2, other3...reduce_dim1, reduce_dim2, reduce_dim3...)
        std::vector<size_t> output_shape;
        std::vector<ptrdiff_t> permuted_input_strides; // need to permute strides for reduce tensor (other1, other2, other3...reduce_dim1, reduce_dim2, reduce_dim3...)
        std::vector<ptrdiff_t> output_strides;
        size_t reduce_dim_size; // reduce dim size
        size_t reduce_num; // number of elements to reduce for each output element
        size_t input_size; // total number of input elements
        size_t output_size; // total number of output elements
        static utils::Result<SumInfo> create(
            infiniopTensorDescriptor_t output_desc,
            infiniopTensorDescriptor_t input_desc,
            size_t *dim, 
            size_t dim_size,
            bool keepdim){
                // CHECK_OR_RETURN(output_desc != nullptr && input_desc != nullptr, INFINI_STATUS_NULL_POINTER); 
                // CHECK_OR_RETURN(output_desc->dtype() == input_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
                auto input_shape = input_desc->shape();
                auto input_strides = input_desc->strides();
                size_t input_ndim = input_desc->ndim();
                size_t reduce_num = 1;
                for(size_t i = 0; i < dim_size; i++){
                    reduce_num *= input_shape[dim[i]];
                }
                std::vector<size_t> permute_order;
                for(size_t i = 0; i < input_ndim; i++){
                    if(std::find(dim, dim + dim_size, i) == dim + dim_size){
                        permute_order.push_back(i);
                    }
                }
                for(size_t i = 0; i < dim_size; i++){
                    permute_order.push_back(dim[i]);
                }
                // CHECK_OR_RETURN(input_ndim == permute_order.size(), INFINI_STATUS_BAD_PARAM);
                std::vector<size_t> permuted_input_shape;
                std::vector<ptrdiff_t> permuted_input_strides;
                for(size_t i = 0; i < permute_order.size(); i++){
                    permuted_input_shape.push_back(input_shape[permute_order[i]]);
                    permuted_input_strides.push_back(input_strides[permute_order[i]]);
                }
                return utils::Result<SumInfo>(SumInfo{input_desc->dtype(),
                                                      permuted_input_shape,
                                                      output_desc->shape(), 
                                                      permuted_input_strides, 
                                                      output_desc->strides(), 
                                                      dim_size, 
                                                      reduce_num, 
                                                      input_desc->numel(), 
                                                      output_desc->numel()});
            }
};
}

#endif
