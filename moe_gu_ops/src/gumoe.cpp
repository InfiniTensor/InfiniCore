#include "gu_moe.h" 
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cstdint> 
#include <tuple> 
#include <iostream>
#include <cstdio>

#include "src/nvidia_kernels/nvidia_kernels_moe.h"
#include "infinicore/ops.hpp"
#include "infinirt.h" 
#include "infiniop.h" 
#include "gu_mul.h"
#include "gu_topk_softmax.h" 

#define LOG_ERR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            LOG_ERR("CUDA Error at line %d: %s", __LINE__, cudaGetErrorString(err)); \
        } \
    } while (0)

namespace infinicore::nn {

GuMoeTopkRounter::GuMoeTopkRounter(int num_experts, int hidden_dim, int top_k, bool norm_topk_prob, const DataType &dtype, const Device &device)
    : top_k_(top_k), num_experts_(num_experts), hidden_dim_(hidden_dim), norm_topk_prob_(norm_topk_prob) {
    infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
    infiniopCreateHandle(&this->handle_);
    INFINICORE_NN_PARAMETER_INIT(weight, ({ {static_cast<size_t>(num_experts_), static_cast<size_t>(hidden_dim_)}, dtype, device }));
}
GuMoeTopkRounter::~GuMoeTopkRounter() { if (handle_) infiniopDestroyHandle(handle_); }
std::pair<Tensor, Tensor> GuMoeTopkRounter::forward(const Tensor &hidden_states) const {
    size_t total_tokens = hidden_states->numel() / hidden_dim_;
    Tensor flattened = hidden_states->view({total_tokens, static_cast<size_t>(hidden_dim_)});
    Tensor logits = infinicore::op::linear(flattened, weight_, std::nullopt);
    auto [val, idx] = infinicore::op::topk_softmax(logits, top_k_, norm_topk_prob_, this->handle_);
    return {val, idx};
}

GuMoeExperts::GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, const DataType& dtype, const Device& device)
    : num_experts_(num_experts), hidden_dim_(hidden_dim), intermediate_dim_(intermediate_dim), device_(device) {
    infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
    infiniopCreateHandle(&this->handle_);
    INFINICORE_NN_PARAMETER_INIT(gate_up_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(2 * intermediate_dim), static_cast<size_t>(hidden_dim)}, dtype, device }));
    INFINICORE_NN_PARAMETER_INIT(down_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim), static_cast<size_t>(intermediate_dim)}, dtype, device }));
}
GuMoeExperts::~GuMoeExperts() { if (handle_) infiniopDestroyHandle(handle_); }

Tensor GuMoeExperts::forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const {
    Device device = hidden_states->device();
    cudaStream_t stream = 0; 

    size_t num_tokens = hidden_states->numel() / hidden_dim_;
    int top_k = top_k_index->shape()[1];
    size_t expanded_size = num_tokens * top_k;

    Tensor indices_i32 = Tensor::empty(top_k_index->shape(), DataType::I32, device);
    size_t num_elements = top_k_index->numel();
    
    // 简单的启发式检查：如果 Dtype 是 5, 6, 7(I64), 0(F32)
    int type_id = (int)top_k_index->dtype();
    
    if (type_id == 7) { 
        Tensor cpu_indices = top_k_index->to(Device(Device::Type::CPU));
        std::vector<int32_t> vec_i32(num_elements);
        const int64_t* ptr = (const int64_t*)cpu_indices->data();
        for(size_t i=0; i<num_elements; ++i) vec_i32[i] = static_cast<int32_t>(ptr[i]);
        CHECK_CUDA(cudaMemcpyAsync(indices_i32->data(), vec_i32.data(), vec_i32.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    } 
    else {
        CHECK_CUDA(cudaMemcpyAsync(indices_i32->data(), top_k_index->data(), num_elements * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    }

    Tensor expert_counts = Tensor::zeros({(size_t)num_experts_ + 1}, DataType::I32, device);
    Tensor expert_offsets = Tensor::zeros({(size_t)num_experts_ + 1}, DataType::I32, device);
    Tensor sorted_input = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
    Tensor sorted_output = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
    Tensor sorted_row_map = Tensor::empty({expanded_size}, DataType::I32, device);
    Tensor sorted_weights = Tensor::empty({expanded_size}, DataType::F32, device);
    Tensor final_output = Tensor::zeros(hidden_states->shape(), DataType::F32, device);

    // 2. 排序
    launch_moe_sort(
        (int32_t*)indices_i32->data(), 
        (int32_t*)expert_counts->data(), 
        (int32_t*)expert_offsets->data(), 
        (int)num_tokens, top_k, num_experts_, stream
    );
    
    launch_moe_permute(
        (float*)hidden_states->data(), 
        (int32_t*)indices_i32->data(), 
        (float*)top_k_values->data(), 
        (int32_t*)expert_offsets->data(),
        (float*)sorted_input->data(), 
        (int32_t*)sorted_row_map->data(), 
        (float*)sorted_weights->data(),
        (int32_t*)expert_counts->data(), 
        (int)num_tokens, top_k, hidden_dim_, num_experts_, stream
    );

    // 3. 拷回 Offsets 供循环使用
    std::vector<int32_t> h_offsets(num_experts_ + 1);
    CHECK_CUDA(cudaMemcpyAsync(h_offsets.data(), expert_offsets->data(), sizeof(int32_t) * (num_experts_ + 1), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream)); // 必须同步等 offsets 回来

    // 4. 计算 Loop
    for (int e = 0; e < num_experts_; ++e) {
        int start_idx = h_offsets[e];
        int count = h_offsets[e+1] - start_idx;
        
        if (count <= 0) continue;

        { 
            Tensor expert_in = sorted_input->narrow({{0, (size_t)start_idx, (size_t)count}});
            Tensor w_gate_up = gate_up_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)(2*intermediate_dim_), (size_t)hidden_dim_});
            Tensor w_down = down_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)hidden_dim_, (size_t)intermediate_dim_});

            Tensor gate_up_out = infinicore::op::linear(expert_in, w_gate_up, std::nullopt);
            Tensor gate = gate_up_out->narrow({{1, 0, (size_t)intermediate_dim_}});
            Tensor up = gate_up_out->narrow({{1, (size_t)intermediate_dim_, (size_t)intermediate_dim_}});
            
            Tensor ffn_inner = infinicore::op::mul(infinicore::op::silu(gate), up, this->handle_);
            Tensor expert_res = infinicore::op::linear(ffn_inner, w_down, std::nullopt);

            CHECK_CUDA(cudaMemcpyAsync((float*)sorted_output->data() + start_idx * hidden_dim_, (float*)expert_res->data(), (size_t)count * hidden_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
    }

    launch_moe_reduce((float*)sorted_output->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(), (float*)final_output->data(), (int)num_tokens, top_k, hidden_dim_, stream);
    
    // 如果是最后一步，通常不需要显式同步，除非后续逻辑需要
    // cudaStreamSynchronize(stream); 
    return final_output;
}

GuMoeSparseMoeBlock::GuMoeSparseMoeBlock(int num_experts, int hidden_dim, int intermediate_dim, int top_k, bool norm_topk, const DataType& dtype, const Device& device) {
    router_ = register_module<GuMoeTopkRounter>("router", num_experts, hidden_dim, top_k, norm_topk, dtype, device);
    experts_ = register_module<GuMoeExperts>("experts", num_experts, hidden_dim, intermediate_dim, dtype, device);
}
Tensor GuMoeSparseMoeBlock::forward(const Tensor& hidden_states) {
    size_t total_tokens = hidden_states->numel() / (hidden_states->shape().back());
    Tensor hidden_states_reshaped = hidden_states->view({total_tokens, hidden_states->shape().back()});
    auto [routing_weights, selected_experts] = router_->forward(hidden_states_reshaped);
    Tensor final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);
    return final_hidden_states->view(hidden_states->shape());
}

} // namespace infinicore::nn