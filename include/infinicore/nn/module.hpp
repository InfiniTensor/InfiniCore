#pragma once

#include "parameter.hpp"

#include <unordered_map>

namespace infinicore::nn {
class Module {
public:
    const std::unordered_map<std::string, Parameter> &state_dict() const;

    void load_state_dict(const std::unordered_map<std::string, Tensor> &_state_dict);

    void load_parameter(const std::string &name, const Tensor &param);

    void load_parameter_from_blob(const std::string &name, const void *data);

    Tensor register_parameter(const std::string &name, Parameter param);

    template <typename M>
    std::shared_ptr<M> add_module(const std::string &name, std::shared_ptr<M> submodule) {
        submodules_[name] = submodule;
        for (auto &p : submodule->parameters_) {
            parameters_[name + "." + p.first] = p.second;
        }
        return submodule;
    }

    template <typename M, typename... Args>
    std::shared_ptr<M> register_module(const std::string &name, Args &&...args) {
        auto submodule = std::make_shared<M>(std::forward<Args>(args)...);
        return add_module(name, submodule);
    }

    template <typename M, typename... Args>
    std::vector<std::shared_ptr<M>> register_modules(size_t layers, const std::string &name, Args &&...args) {
        auto submodules = std::vector<std::shared_ptr<M>>(layers);
        for (size_t i = 0; i < layers; i++) {
            register_module<M>(name + "." + std::to_string(i), std::forward<Args>(args)...);
        }
        return submodules;
    }

protected:
    Device device_;
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;
    std::unordered_map<std::string, Parameter> parameters_;
};
} // namespace infinicore::nn