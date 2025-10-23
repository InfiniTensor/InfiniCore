#include "infinicore/nn/module.hpp"

namespace infinicore::nn {
const std::unordered_map<std::string, Parameter> &Module::state_dict() const {
    return parameters_;
}

void Module::load_state_dict(const std::unordered_map<std::string, Tensor> &_state_dict) {
    for (auto &p : parameters_) {
        load_parameter(p.first, p.second);
    }
}

void Module::load_parameter(const std::string &name, const Tensor &param) {
    parameters_[name]->copy_from(param);
}

void Module::load_parameter_from_blob(const std::string &name, const void *data) {
    auto param = parameters_[name];
    param.load_blob(data);
}

Tensor Module::register_parameter(const std::string &name, Parameter param) {
    parameters_[name] = param;
    return param;
}

} // namespace infinicore::nn
