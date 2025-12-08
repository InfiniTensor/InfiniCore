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
void Sum::execute(Tensor output, Tensor input, std::vector<size_t> dim = None, bool keepdim = false) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Sum implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, dim, keepdim);
}

// 确定下构建input的时候是不是已经按strides构建了？所以这里不用处理strides
// todo 在这里确定output的形状, dim已经在hpp中转化为了vector<size_t>, 确保dim已经sort过了
// 那么是不是后续都用不到keepdim了？？todo ifso : 去除掉后续的keepdim参数
Tensor sum(Tensor input, std::vector<size_t> dim, bool keepdim=false) {
    auto in_shape = input->shape();
    std::vector<size_t> out_shape;
    std::sort(dim.begin(), dim.end()); // 是不是在test base确定kwargs时已经sort过了？
    // 是不是要用标记数组给dim去重？
    if (dim[0] < 0 || dim.size() > in_shape.size()){
        throw std::invalid_argument("dim is out of range");
    }
    // 寻求简化下面的代码
    if(keepdim){
        size_t j = 0;
        for(size_t i = 0; i < in_shape.size() && j < dim.size(); i++){
            if(i < dim.size() && dim[j] != i){
                out_shape.push_back(i);
            } else {
                out_shape.push_back(1);
                j++;
            }
        }
    } else {
        size_t j = 0;
        for(size_t i = 0; i < in_shape.size() && j < dim.size(); i++){
            if(i < dim.size() && dim[j] != i){
                out_shape.push_back(in_shape[i]);
            } else {
                j++;
            }
        }
    }

    // auto out_shape = in_shape.copy(); // 拷贝一份in_shape 不过要结合
    // std::sort(remove_dims.begin(), remove_dims.end(), greater<size_t>());
    // for (auto d : remove_dims) {
    //     in_shape.erase(in_shape.begin() + d);
    // }
    
    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    sum_(output, input, dim, keepdim); // todo sum_ 即 具体的execute函数中，再通过keepdim决定output的shape？ 还是说output的shape就要在这里确定？
    return output;
}



void sum_(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    Sum::execute(output, input, dim, keepdim);
}
} // namespace infinicore::op
