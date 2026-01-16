#ifndef GU_MOE_H
#define GU_MOE_H

#include <vector>
#include <string>
#include <memory>
#include "infinicore/tensor.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops.hpp"

namespace infinicore::nn {

// Router
class GuMoeTopkRounter : public Module {
private:
    int top_k_;
    int num_experts_;
    int hidden_dim_;
    bool norm_topk_prob_;
    Parameter weight_; 
    infiniopHandle_t handle_ = nullptr; 
public:
    GuMoeTopkRounter(int num_experts, int hidden_dim, int top_k, bool norm_topk_prob,
                     const DataType &dtype, const Device &device);
    ~GuMoeTopkRounter();
    void set_weight(Tensor w); // 设置路由权重
    std::pair<Tensor, Tensor> forward(const Tensor &hidden_states) const;
};

// Experts
class GuMoeExperts : public Module {
private:
    int num_experts_;
    int hidden_dim_;
    int intermediate_dim_;
    Parameter gate_up_proj_; 
    Parameter down_proj_;
    infiniopHandle_t handle_ = nullptr;
    Device device_;

public:
    GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, 
                 const DataType& dtype, const Device& device);
    ~GuMoeExperts();
    void set_weights(Tensor gate_up, Tensor down); // 设置专家权重
    Tensor forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const;
};

// Block (MoE 整体)
class GuMoeSparseMoeBlock : public Module {
private:
    std::shared_ptr<GuMoeTopkRounter> router_;
    std::shared_ptr<GuMoeExperts> experts_;

public:
    GuMoeSparseMoeBlock(int num_experts, int hidden_dim, int intermediate_dim, 
                        int top_k, bool norm_topk, 
                        const DataType& dtype, const Device& device);

    // ✅ 统一设置权重的接口
    void set_weights(Parameter gate_up, Parameter down, Parameter router_weight);

    // ✅ 关键：Forward 只需要 Input (Router 在内部计算)
    Tensor forward(const Tensor& hidden_states);
};

} // namespace
#endif