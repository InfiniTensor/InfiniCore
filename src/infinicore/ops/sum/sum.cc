#include "infinicore/ops/sum.hpp"

#include "../../utils.hpp"
#include <vector>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Sum::schema> &Sum::dispatcher() {
    static common::OpDispatcher<Sum::schema> dispatcher_;
    return dispatcher_;
};
// todo 12.8 完成这里的代码
void Sum::execute(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(input->device());
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Sum implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, dim, keepdim);
}


Tensor sum(Tensor input, std::vector<size_t> dim, bool keepdim) {
    auto in_shape = input->shape();
    std::vector<size_t> out_shape;
    // 是不是要用标记数组给dim去重？update: 不用去重 ,  因为torch中sum的dim参数不能重复，可以在输入阶段加以判断
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
    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    sum_(output, input, dim, keepdim); // todo sum_ 即 具体的execute函数中，再通过keepdim决定output的shape？ 还是说output的shape就要在这里确定？
    return output;
}



void sum_(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    Sum::execute(output, input, dim, keepdim);
}
} // namespace infinicore::op
