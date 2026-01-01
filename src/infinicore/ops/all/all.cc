#include "infinicore/ops/all.hpp"

#include "../../utils.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
namespace infinicore::op {

common::OpDispatcher<All::schema> &All::dispatcher() {
    static common::OpDispatcher<All::schema> dispatcher_;
    return dispatcher_;
};
// todo 12.8 完成这里的代码
void All::execute(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(input->device());
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No All implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, dim, keepdim);
}


Tensor all(Tensor input, std::vector<size_t> dim, bool keepdim) {
    auto in_shape = input->shape();
    std::vector<size_t> out_shape;
    if (dim.empty()) {
        // dim 为空时，对所有维度求和
        for (size_t i = 0; i < in_shape.size(); i++) {
            dim.push_back(i);
        }
    }
    std::sort(dim.begin(), dim.end());
    if (dim.size() == in_shape.size() && !keepdim) {
        out_shape = {};  // 标量，0维tensor
    } else {
        if(keepdim){
            size_t j = 0;
            for(size_t i = 0; i < in_shape.size(); i++){
                if(j < dim.size() && dim[j] == i){
                    out_shape.push_back(1);
                    j++;
                } else {
                    out_shape.push_back(in_shape[i]);
                }
            }
        } else {
            size_t j = 0;
            for(size_t i = 0; i < in_shape.size(); i++){
                if(j < dim.size() && dim[j] == i){
                    j++;
                } else {
                    out_shape.push_back(in_shape[i]);
                }
            }
        }
    }
    // auto output = Tensor::empty(out_shape, INFINI_DTYPE_BOOL, input->device());
    auto output = Tensor::empty(out_shape, DataType::BOOL, input->device());
    all_(output, input, dim, keepdim); 
    return output;
}



void all_(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    // std::cout << "all_ output: " << output->shape() << std::endl;
    std::cout << "++++++++++++++output++++++++++++++"<< std::endl;
    for(int i = 0; i < output->ndim(); i++){
        std::cout << output->shape()[i] << "|shape|";
    }
    std::cout << std::endl;
    for(int i = 0; i < output->ndim(); i++){
        std::cout << output->strides()[i] << "|strides|";
    }
    std::cout << "--------------output---------------"<< std::endl;
    All::execute(output, input, dim, keepdim);
}
} // namespace infinicore::op
