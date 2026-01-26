GuMoeExperts::GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, const DataType& dtype, const Device& device)

: num_experts_(num_experts), hidden_dim_(hidden_dim), intermediate_dim_(intermediate_dim), device_(device) {

infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());

infiniopCreateHandle(&this->handle_);

INFINICORE_NN_PARAMETER_INIT(gate_up_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(2 * intermediate_dim), static_cast<size_t>(hidden_dim)}, dtype, device }));

INFINICORE_NN_PARAMETER_INIT(down_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim), static_cast<size_t>(intermediate_dim)}, dtype, device }));

}

GuMoeExperts::~GuMoeExperts() { if (handle_) infiniopDestroyHandle(handle_); }



Tensor GuMoeExperts::forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const {

if (hidden_states->dtype() != DataType::F32) throw std::runtime_error("F32 only");



Device gpu = hidden_states->device();

Device cpu(Device::Type::CPU);



Tensor cpu_indices = top_k_index->to(cpu);

Tensor cpu_values = top_k_values->to(cpu);

Tensor cpu_hidden = hidden_states->to(cpu);


Tensor final_cpu_states = Tensor::zeros(hidden_states->shape(), hidden_states->dtype(), cpu);

std::memset(final_cpu_states->data(), 0, final_cpu_states->numel() * sizeof(float));



size_t total_tokens = hidden_states->numel() / hidden_dim_;

int top_k = top_k_index->shape()[1];



struct Task { int token_idx; int rank_idx; };

std::vector<std::vector<Task>> buckets(num_experts_);

const void* raw_idx = cpu_indices->data();


bool is_i32 = (cpu_indices->dtype() == DataType::I32);

const float* all_vals = (const float*)cpu_values->data();



// 路由信息打印 (保持你之前的)

static bool debug_printed = false;

if (!debug_printed) {

std::cout << "\n[C++ Debug Info]" << std::endl;

std::cout << "Token 0 Selected Experts: [";

for(int k=0; k<top_k; ++k) {

int64_t val = is_i32 ? (int64_t)((const int32_t*)raw_idx)[k] : ((const int64_t*)raw_idx)[k];

std::cout << val << (k == top_k - 1 ? "" : ", ");

}

std::cout << "]" << std::endl;

std::cout << "Token 0 Expert Weights: [";

for(int k=0; k<top_k; ++k) {

std::cout << all_vals[k] << (k == top_k - 1 ? "" : ", ");

}

std::cout << "]" << std::endl;

}



for (size_t i = 0; i < total_tokens; ++i) {

for (size_t k = 0; k < static_cast<size_t>(top_k); ++k) {

int64_t val = is_i32 ? (int64_t)((const int32_t*)raw_idx)[i*top_k+k] : ((const int64_t*)raw_idx)[i*top_k+k];

int eid = (int)val;

if (eid >= 0 && eid < num_experts_) buckets[eid].push_back({(int)i, (int)k});

}

}



infinirtSetDevice((infiniDevice_t)device_.getType(), device_.getIndex());



for (int e = 0; e < num_experts_; ++e) {

if (buckets[e].empty()) continue;

size_t n = buckets[e].size();



std::vector<int> t_idx(n);

std::vector<float> t_w(n);


// 查找 Token 0 在当前 bucket 中的位置

int local_token0_idx = -1;



for(size_t i=0; i<n; ++i) {

t_idx[i] = buckets[e][i].token_idx;

if (t_idx[i] == 0) local_token0_idx = (int)i; // 标记位置

t_w[i] = all_vals[t_idx[i]*top_k + buckets[e][i].rank_idx];

}



Tensor cpu_in = Tensor::empty({n, (size_t)hidden_dim_}, hidden_states->dtype(), cpu);

cpu_gather((float*)cpu_in->data(), (const float*)cpu_hidden->data(), t_idx, hidden_dim_);

Tensor gpu_in = cpu_in->to(gpu);



Tensor w_gate_up = gate_up_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)(2*intermediate_dim_), (size_t)hidden_dim_});

Tensor gate_up_out = infinicore::op::linear(gpu_in, w_gate_up, std::nullopt);



// =====================================================================

// 🕵️‍♂️ [新增] FFN 中间值探针 (Gate/Up)

// =====================================================================

static bool ffn_debug_printed = false;

if (!ffn_debug_printed && local_token0_idx != -1) {

std::cout << "\n[C++ FFN Internal Debug] Expert " << e << " processing Token 0" << std::endl;


// 拷回 CPU

Tensor debug_tensor = gate_up_out->to(cpu);

const float* ptr = (const float*)debug_tensor->data();


// 定位到 Token 0 的那一行数据

// shape: [n, 2 * intermediate_dim]

size_t row_offset = local_token0_idx * (2 * intermediate_dim_);

const float* token0_row = ptr + row_offset;



// 打印前半部分 (C++ 认为是 Gate)

std::cout << " C++ First Half (Gate?): [";

for(int j=0; j<5; ++j) std::cout << token0_row[j] << ", ";

std::cout << "...]" << std::endl;



// 打印后半部分 (C++ 认为是 Up)

size_t mid = intermediate_dim_;

std::cout << " C++ Second Half (Up?): [";

for(int j=0; j<5; ++j) std::cout << token0_row[mid+j] << ", ";

std::cout << "...]" << std::endl;


ffn_debug_printed = true;

}

// =====================================================================



Tensor gate = gate_up_out->narrow({{1, 0, (size_t)intermediate_dim_}});

Tensor up = gate_up_out->narrow({{1, (size_t)intermediate_dim_, (size_t)intermediate_dim_}});

Tensor ffn_inner = infinicore::op::mul(infinicore::op::silu(gate), up, this->handle_);


Tensor w_down = down_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)hidden_dim_, (size_t)intermediate_dim_});

Tensor gpu_res = infinicore::op::linear(ffn_inner, w_down, std::nullopt);


Tensor cpu_res = gpu_res->to(cpu);



infinirtDeviceSynchronize();



cpu_index_add_scale((float*)final_cpu_states->data(),

(const float*)cpu_res->data(),

t_idx, t_w, hidden_dim_,

total_tokens);

}


if (!debug_printed) debug_printed = true; // 防止漏打导致多次



return final_cpu_states->to(gpu);

}

// #include "gu_moe.h" 

// #include <cstring>
// #include <stdexcept>
// #include <vector>
// #include <cstdint> 
// #include <iostream> 
// #include <iomanip>  
// #include <cmath>    

// #include "src/nvidia_kernels/nvidia_kernels_moe.h"
// #include "infinicore/ops.hpp"
// #include "infinicore/ops/linear.hpp"
// #include "infinirt.h" 
// #include "infiniop.h" 
// #include "gu_mul.h"
// #include "gu_topk_softmax.h"

// namespace infinicore::nn {

// namespace {

// void debug_tensor(const std::string& name, const Tensor& t, int count=5) {
//     Device cpu(Device::Type::CPU);
//     Tensor c = t->to(cpu);
//     if (c->dtype() == DataType::F32) {
//         const float* ptr = reinterpret_cast<const float*>(c->data());
//         float min_v = 1e30, max_v = -1e30;
//         double sum = 0;
//         for(size_t i=0; i<c->numel(); ++i) {
//             float v = ptr[i];
//             if(v < min_v) min_v = v;
//             if(v > max_v) max_v = v;
//             sum += std::abs(v);
//         }
//         std::cout << "[DEBUG] " << name << " | Min: " << min_v << " | Max: " << max_v 
//                   << " | MeanAbs: " << (sum / c->numel()) << std::endl;
//     }
// }

// void cpu_gather(float* dest, const float* src, const std::vector<int>& indices, int hidden_dim) {
//     for (size_t i = 0; i < indices.size(); ++i) {
//         int row = indices[i];
//         std::memcpy(dest + i * hidden_dim, src + row * hidden_dim, hidden_dim * sizeof(float));
//     }
// }

// void cpu_index_add_scale(float* dest, const float* src, 
//                          const std::vector<int>& indices, 
//                          const std::vector<float>& weights, 
//                          int hidden_dim, 
//                          size_t total_rows) { 
//     for (size_t i = 0; i < indices.size(); ++i) {
//         int row = indices[i];
//         if (row < 0 || row >= (int)total_rows) continue; 
//         float w = weights[i];
//         float* d_row = dest + row * hidden_dim;
//         const float* s_row = src + i * hidden_dim;
//         for (int j = 0; j < hidden_dim; ++j) {
//             d_row[j] += s_row[j] * w;
//         }
//     }
// }

// } // namespace anonymous

// // ... GuMoeTopkRounter ...
// GuMoeTopkRounter::GuMoeTopkRounter(int num_experts, int hidden_dim, int top_k, bool norm_topk_prob, const DataType &dtype, const Device &device)
//     : top_k_(top_k), num_experts_(num_experts), hidden_dim_(hidden_dim), norm_topk_prob_(norm_topk_prob) {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(weight, ({ {static_cast<size_t>(num_experts_), static_cast<size_t>(hidden_dim_)}, dtype, device }));
// }
// GuMoeTopkRounter::~GuMoeTopkRounter() { if (handle_) infiniopDestroyHandle(handle_); }

// std::pair<Tensor, Tensor> GuMoeTopkRounter::forward(const Tensor &hidden_states) const {
//     size_t total_tokens = hidden_states->numel() / hidden_dim_;
//     Tensor flattened = hidden_states->view({total_tokens, static_cast<size_t>(hidden_dim_)});
//     Tensor logits = infinicore::op::linear(flattened, weight_, std::nullopt);
//     auto [val, idx] = infinicore::op::topk_softmax(logits, top_k_, norm_topk_prob_, this->handle_);
//     return {val, idx};
// }

// // ... GuMoeExperts ...
// GuMoeExperts::GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, const DataType& dtype, const Device& device)
//     : num_experts_(num_experts), hidden_dim_(hidden_dim), intermediate_dim_(intermediate_dim), device_(device) {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(gate_up_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(2 * intermediate_dim), static_cast<size_t>(hidden_dim)}, dtype, device }));
//     INFINICORE_NN_PARAMETER_INIT(down_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim), static_cast<size_t>(intermediate_dim)}, dtype, device }));
// }
// GuMoeExperts::~GuMoeExperts() { if (handle_) infiniopDestroyHandle(handle_); }

// Tensor GuMoeExperts::forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const {
//     if (hidden_states->dtype() != DataType::F32) throw std::runtime_error("F32 only");

//     // 0. 上下文准备
//     Device device = hidden_states->device();
//     // 假设使用默认流 0。如果 infiniop 支持获取流，建议使用 context::getStream()
//     cudaStream_t stream = 0;

//     size_t num_tokens = hidden_states->numel() / hidden_dim_;
//     int top_k = top_k_index->shape()[1];
//     size_t expanded_size = num_tokens * top_k;

//     // 1. 分配 GPU 显存 (Workspace)
//     // 工业级优化点：这里的 Tensor::zeros/empty 每次 forward 都会申请显存。
//     // 如果追求极致性能，建议在类里维护一个缓存池 (Tensor workspace_)。
    
//     // 计数器和偏移量
//     Tensor expert_counts = Tensor::zeros({(size_t)num_experts_}, DataType::I32, device);
//     Tensor expert_offsets = Tensor::zeros({(size_t)num_experts_ + 1}, DataType::I32, device);

//     // 中间 buffer (排序后的输入/输出)
//     Tensor sorted_input = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_output = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
    
//     // 辅助信息 (Row Map 和 Weights)
//     Tensor sorted_row_map = Tensor::empty({expanded_size}, DataType::I32, device);
//     Tensor sorted_weights = Tensor::empty({expanded_size}, DataType::F32, device);
    
//     // 最终输出 (必须初始化为 0，因为 Reduce 是累加)
//     Tensor final_output = Tensor::zeros(hidden_states->shape(), DataType::F32, device);

//     // 获取裸指针
//     float* d_input     = (float*)hidden_states->data();
//     int32_t* d_indices = (int32_t*)top_k_index->data();
//     float* d_values    = (float*)top_k_values->data();
    
//     int32_t* d_counts  = (int32_t*)expert_counts->data();
//     int32_t* d_offsets = (int32_t*)expert_offsets->data();

//     // ======================================================================
//     // Phase 1: 数据重排 (GPU Sort & Permute)
//     // 彻底取代原来的 CPU bucket 和 cpu_gather
//     // ======================================================================
    
//     // 1.1 排序：计算每个专家的 Token 数量和偏移量
//     launch_moe_sort(
//         d_indices, d_counts, d_offsets, 
//         num_tokens, top_k, num_experts_, 
//         stream
//     );

//     // 1.2 搬运：将 Input 和 Weights 按照专家顺序连续排列到 sorted_input/sorted_weights
//     // 注意：复用 expert_counts 作为 running_counters (内部会自动清零)
//     launch_moe_permute(
//         d_input, 
//         d_indices, 
//         d_values, 
//         d_offsets,
//         (float*)sorted_input->data(), 
//         (int32_t*)sorted_row_map->data(),
//         (float*)sorted_weights->data(),
//         d_counts, 
//         num_tokens, top_k, hidden_dim_, num_experts_, 
//         stream
//     );

//     // ======================================================================
//     // Phase 2: 计算 (GPU Loop)
//     // 这里的循环仅用于发射 Kernel，数据全程在 GPU 上，没有拷贝开销
//     // ======================================================================

//     // 将 Offsets 拷回 CPU，以便 CPU 知道如何对 sorted_input 进行切片
//     std::vector<int32_t> h_offsets(num_experts_ + 1);
//     cudaMemcpyAsync(h_offsets.data(), d_offsets, sizeof(int32_t) * (num_experts_ + 1), cudaMemcpyDeviceToHost, stream);
//     cudaStreamSynchronize(stream); // 等待 Offset 拷贝完成

//     for (int e = 0; e < num_experts_; ++e) {
//         int start_idx = h_offsets[e];
//         int count = h_offsets[e+1] - start_idx;

//         // 如果该专家没有分配到 Token，跳过
//         if (count == 0) continue;

//         // A. 切片 (Slicing - Zero Copy)
//         // 这里的 narrow 只是创建 View，不发生数据搬运
//         // 切出属于当前专家的输入数据
//         Tensor expert_in = sorted_input->narrow({{0, (size_t)start_idx, (size_t)count}});

//         // 切出当前专家的权重
//         Tensor w_gate_up = gate_up_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)(2*intermediate_dim_), (size_t)hidden_dim_});
//         Tensor w_down = down_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)hidden_dim_, (size_t)intermediate_dim_});

//         // B. 计算 (Computation - All on GPU)
//         // 1. Linear: Input * GateUp
//         Tensor gate_up_out = infinicore::op::linear(expert_in, w_gate_up, std::nullopt);

//         // 2. Activation: SiLU(Gate) * Up
//         Tensor gate = gate_up_out->narrow({{1, 0, (size_t)intermediate_dim_}});
//         Tensor up = gate_up_out->narrow({{1, (size_t)intermediate_dim_, (size_t)intermediate_dim_}});
        
//         // FFN Inner
//         Tensor ffn_inner = infinicore::op::mul(infinicore::op::silu(gate), up, this->handle_);

//         // 3. Linear: Inner * Down
//         Tensor expert_res = infinicore::op::linear(ffn_inner, w_down, std::nullopt);

//         // C. 写回大 Buffer (Scatter back to sorted_output)
//         // infiniop::linear 返回的是新分配的 Tensor，我们需要把它拷贝回 sorted_output 的对应位置
//         // 这一步是 Device-to-Device Copy，速度极快
        
//         float* dst_ptr = (float*)sorted_output->data() + start_idx * hidden_dim_;
//         const float* src_ptr = (const float*)expert_res->data();
//         size_t bytes = count * hidden_dim_ * sizeof(float);

//         cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream);
//     }

//     // ======================================================================
//     // Phase 3: 还原 (GPU Reduce)
//     // 使用 sorted_row_map 和 sorted_weights 将结果加权累加回 final_output
//     // ======================================================================
    
//     launch_moe_reduce(
//         (float*)sorted_output->data(),
//         (int32_t*)sorted_row_map->data(),
//         (float*)sorted_weights->data(),
//         (float*)final_output->data(),
//         num_tokens, top_k, hidden_dim_, 
//         stream
//     );

//     return final_output;
// }

// // ... 保持不变 ...
// GuMoeSparseMoeBlock::GuMoeSparseMoeBlock(int num_experts, int hidden_dim, int intermediate_dim, 
//                                          int top_k, bool norm_topk, 
//                                          const DataType& dtype, const Device& device) {
//     router_ = register_module<GuMoeTopkRounter>("router", num_experts, hidden_dim, top_k, norm_topk, dtype, device);
//     experts_ = register_module<GuMoeExperts>("experts", num_experts, hidden_dim, intermediate_dim, dtype, device);
// }
// Tensor GuMoeSparseMoeBlock::forward(const Tensor& hidden_states) {
//     auto input_shape = hidden_states->shape();
//     size_t batch_size = input_shape[0];
//     size_t seq_len = input_shape[1];
//     size_t hidden_dim = input_shape[2];
//     size_t total_tokens = hidden_states->numel() / hidden_dim;
//     Tensor hidden_states_reshaped = hidden_states->view({total_tokens, hidden_dim});
//     auto [routing_weights, selected_experts] = router_->forward(hidden_states_reshaped);
//     Tensor final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);
//     return final_hidden_states->view({batch_size, seq_len, hidden_dim});
// }

// } // namespace

// #include "gu_moe.h" 

// #include <cstring>
// #include <stdexcept>
// #include <vector>
// #include <cstdint> 
// #include <iostream> 
// #include <iomanip>  
// #include <cmath>
// #include <tuple> // 补充: 为了 std::get, std::tuple

// // 确保包含你项目中实际存在的头文件
// #include "src/nvidia_kernels/nvidia_kernels_moe.h"
// #include "infinicore/ops.hpp"
// // #include "infinicore/ops/linear.hpp" // 如果 ops.hpp 已包含，可注释
// #include "infinirt.h" 
// #include "infiniop.h" 
// #include "gu_mul.h"
// // #include "gu_mul.h" // 如果不需要可注释
// #include "gu_topk_softmax.h" // 确保这个文件存在

// namespace infinicore::nn {

// namespace {

// void debug_tensor(const std::string& name, const Tensor& t, int count=5) {
//     Device cpu(Device::Type::CPU);
//     Tensor c = t->to(cpu);
//     if (c->dtype() == DataType::F32) {
//         const float* ptr = reinterpret_cast<const float*>(c->data());
//         float min_v = 1e30, max_v = -1e30;
//         double sum = 0;
//         for(size_t i=0; i<c->numel(); ++i) {
//             float v = ptr[i];
//             if(v < min_v) min_v = v;
//             if(v > max_v) max_v = v;
//             sum += std::abs(v);
//         }
//         std::cout << "[DEBUG] " << name << " | Min: " << min_v << " | Max: " << max_v 
//                   << " | MeanAbs: " << (sum / c->numel()) << std::endl;
//     }
// }

// } // namespace anonymous

// // ==========================================
// // GuMoeTopkRounter 实现
// // ==========================================

// GuMoeTopkRounter::GuMoeTopkRounter(int num_experts, int hidden_dim, int top_k, bool norm_topk_prob, const DataType &dtype, const Device &device)
//     : top_k_(top_k), 
//       num_experts_(num_experts), 
//       hidden_dim_(hidden_dim), 
//       norm_topk_prob_(norm_topk_prob)
// {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     // 初始化权重，假设宏 INFINICORE_NN_PARAMETER_INIT 会处理赋值
//     INFINICORE_NN_PARAMETER_INIT(weight, ({ {static_cast<size_t>(num_experts_), static_cast<size_t>(hidden_dim_)}, dtype, device }));
// }

// GuMoeTopkRounter::~GuMoeTopkRounter() { 
//     if (handle_) infiniopDestroyHandle(handle_); 
// }

// std::pair<Tensor, Tensor> GuMoeTopkRounter::forward(const Tensor &hidden_states) const {
//     size_t total_tokens = hidden_states->numel() / hidden_dim_;
//     Tensor flattened = hidden_states->view({total_tokens, static_cast<size_t>(hidden_dim_)});
    
//     Tensor logits = infinicore::op::linear(flattened, weight_, std::nullopt);
    
//     auto [val, idx] = infinicore::op::topk_softmax(logits, top_k_, norm_topk_prob_, this->handle_);
    
//     return {val, idx};
// }

// // ==========================================
// // GuMoeExperts 实现
// // ==========================================

// GuMoeExperts::GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, const DataType& dtype, const Device& device)
//     : num_experts_(num_experts), 
//       hidden_dim_(hidden_dim), 
//       intermediate_dim_(intermediate_dim), 
//       device_(device) 
// {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(gate_up_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(2 * intermediate_dim), static_cast<size_t>(hidden_dim)}, dtype, device }));
//     INFINICORE_NN_PARAMETER_INIT(down_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim), static_cast<size_t>(intermediate_dim)}, dtype, device }));
// }

// GuMoeExperts::~GuMoeExperts() { 
//     if (handle_) infiniopDestroyHandle(handle_); 
// }

// Tensor GuMoeExperts::forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const {
//     if (hidden_states->dtype() != DataType::F32) throw std::runtime_error("F32 only");
    
//     // 0. 上下文准备
//     Device device = hidden_states->device();
//     cudaStream_t stream = 0; // 默认流

//     size_t num_tokens = hidden_states->numel() / hidden_dim_;
//     // 假设 top_k_index shape 是 [num_tokens, top_k]
//     int top_k = top_k_index->shape()[1];
//     size_t expanded_size = num_tokens * top_k;

//     // 1. 分配 GPU 显存 (Workspace)
//     Tensor expert_counts = Tensor::zeros({(size_t)num_experts_}, DataType::I32, device);
//     Tensor expert_offsets = Tensor::zeros({(size_t)num_experts_ + 1}, DataType::I32, device);

//     Tensor sorted_input = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_output = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
    
//     Tensor sorted_row_map = Tensor::empty({expanded_size}, DataType::I32, device);
//     Tensor sorted_weights = Tensor::empty({expanded_size}, DataType::F32, device);
    
//     Tensor final_output = Tensor::zeros(hidden_states->shape(), DataType::F32, device);

//     // 获取裸指针
//     float* d_input     = (float*)hidden_states->data();
//     int32_t* d_indices = (int32_t*)top_k_index->data();
//     float* d_values    = (float*)top_k_values->data();
    
//     int32_t* d_counts  = (int32_t*)expert_counts->data();
//     int32_t* d_offsets = (int32_t*)expert_offsets->data();

//     // ======================================================================
//     // Phase 1: 数据重排 (GPU Sort & Permute)
//     // ======================================================================
    
//     launch_moe_sort(
//         d_indices, d_counts, d_offsets, 
//         num_tokens, top_k, num_experts_, 
//         stream
//     );

//     launch_moe_permute(
//         d_input, 
//         d_indices, 
//         d_values, 
//         d_offsets,
//         (float*)sorted_input->data(), 
//         (int32_t*)sorted_row_map->data(),
//         (float*)sorted_weights->data(),
//         d_counts, 
//         num_tokens, top_k, hidden_dim_, num_experts_, 
//         stream
//     );

//     // ======================================================================
//     // Phase 2: 计算 (GPU Loop)
//     // ======================================================================

//     std::vector<int32_t> h_offsets(num_experts_ + 1);
//     cudaMemcpyAsync(h_offsets.data(), d_offsets, sizeof(int32_t) * (num_experts_ + 1), cudaMemcpyDeviceToHost, stream);
//     cudaStreamSynchronize(stream); 

//     for (int e = 0; e < num_experts_; ++e) {
//         int start_idx = h_offsets[e];
//         int count = h_offsets[e+1] - start_idx;

//         if (count == 0) continue;

//         // A. 切片 (如果 InfiniCore 确实支持 narrow，这里就没问题)
//         // 注意：之前报错说没 narrow，这里保留你的代码。如果再次报错，说明 InfiniCore 只有 slice
//         Tensor expert_in = sorted_input->narrow({{0, (size_t)start_idx, (size_t)count}});

//         Tensor w_gate_up = gate_up_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)(2*intermediate_dim_), (size_t)hidden_dim_});
//         Tensor w_down = down_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)hidden_dim_, (size_t)intermediate_dim_});

//         // B. 计算
//         Tensor gate_up_out = infinicore::op::linear(expert_in, w_gate_up, std::nullopt);

//         Tensor gate = gate_up_out->narrow({{1, 0, (size_t)intermediate_dim_}});
//         Tensor up = gate_up_out->narrow({{1, (size_t)intermediate_dim_, (size_t)intermediate_dim_}});
        
//         // FFN Inner
//         Tensor ffn_inner = infinicore::op::mul(infinicore::op::silu(gate), up, this->handle_);

//         Tensor expert_res = infinicore::op::linear(ffn_inner, w_down, std::nullopt);

//         // C. 写回
//         float* dst_ptr = (float*)sorted_output->data() + start_idx * hidden_dim_;
//         const float* src_ptr = (const float*)expert_res->data();
//         size_t bytes = count * hidden_dim_ * sizeof(float);

//         cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream);
//     }

//     // ======================================================================
//     // Phase 3: 还原 (GPU Reduce)
//     // ======================================================================
    
//     launch_moe_reduce(
//         (float*)sorted_output->data(),
//         (int32_t*)sorted_row_map->data(),
//         (float*)sorted_weights->data(),
//         (float*)final_output->data(),
//         num_tokens, top_k, hidden_dim_, 
//         stream
//     );

//     return final_output;
// }

// GuMoeSparseMoeBlock::GuMoeSparseMoeBlock(int num_experts, int hidden_dim, int intermediate_dim, 
//                                          int top_k, bool norm_topk, 
//                                          const DataType& dtype, const Device& device) {
//     router_ = register_module<GuMoeTopkRounter>("router", num_experts, hidden_dim, top_k, norm_topk, dtype, device);
//     experts_ = register_module<GuMoeExperts>("experts", num_experts, hidden_dim, intermediate_dim, dtype, device);
// }
// Tensor GuMoeSparseMoeBlock::forward(const Tensor& hidden_states) {
//     auto input_shape = hidden_states->shape();
//     size_t batch_size = input_shape[0];
//     size_t seq_len = input_shape[1];
//     size_t hidden_dim = input_shape[2];
//     size_t total_tokens = hidden_states->numel() / hidden_dim;
//     Tensor hidden_states_reshaped = hidden_states->view({total_tokens, hidden_dim});
//     auto [routing_weights, selected_experts] = router_->forward(hidden_states_reshaped);
//     Tensor final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);
//     return final_hidden_states->view({batch_size, seq_len, hidden_dim});
// }

// } // namespace nn

// #include "gu_moe.h" 

// #include <cstring>
// #include <stdexcept>
// #include <vector>
// #include <cstdint> 
// #include <iostream> 
// #include <iomanip>  
// #include <cmath>
// #include <tuple> 

// #include "src/nvidia_kernels/nvidia_kernels_moe.h"
// #include "infinicore/ops.hpp"
// #include "infinirt.h" 
// #include "infiniop.h" 
// #include "gu_mul.h"
// #include "gu_topk_softmax.h" 

// namespace infinicore::nn {

// namespace {

// void debug_tensor(const std::string& name, const Tensor& t, int count=5) {
//     Device cpu(Device::Type::CPU);
//     Tensor c = t->to(cpu);
//     if (c->dtype() == DataType::F32) {
//         const float* ptr = reinterpret_cast<const float*>(c->data());
//         float min_v = 1e30, max_v = -1e30;
//         double sum = 0;
//         for(size_t i=0; i<c->numel(); ++i) {
//             float v = ptr[i];
//             if(v < min_v) min_v = v;
//             if(v > max_v) max_v = v;
//             sum += std::abs(v);
//         }
//         std::cout << "[DEBUG] " << name << " | Min: " << min_v << " | Max: " << max_v 
//                   << " | MeanAbs: " << (sum / c->numel()) << std::endl;
//     }
// }

// } // namespace

// // ==========================================
// // GuMoeTopkRounter 实现
// // ==========================================

// GuMoeTopkRounter::GuMoeTopkRounter(int num_experts, int hidden_dim, int top_k, bool norm_topk_prob, const DataType &dtype, const Device &device)
//     : top_k_(top_k), 
//       num_experts_(num_experts), 
//       hidden_dim_(hidden_dim), 
//       norm_topk_prob_(norm_topk_prob)
// {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(weight, ({ {static_cast<size_t>(num_experts_), static_cast<size_t>(hidden_dim_)}, dtype, device }));
// }

// GuMoeTopkRounter::~GuMoeTopkRounter() { 
//     if (handle_) infiniopDestroyHandle(handle_); 
// }

// std::pair<Tensor, Tensor> GuMoeTopkRounter::forward(const Tensor &hidden_states) const {
//     size_t total_tokens = hidden_states->numel() / hidden_dim_;
//     Tensor flattened = hidden_states->view({total_tokens, static_cast<size_t>(hidden_dim_)});
//     Tensor logits = infinicore::op::linear(flattened, weight_, std::nullopt);
//     auto [val, idx] = infinicore::op::topk_softmax(logits, top_k_, norm_topk_prob_, this->handle_);
//     return {val, idx};
// }

// // ==========================================
// // GuMoeExperts 实现
// // ==========================================

// GuMoeExperts::GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, const DataType& dtype, const Device& device)
//     : num_experts_(num_experts), 
//       hidden_dim_(hidden_dim), 
//       intermediate_dim_(intermediate_dim), 
//       device_(device) 
// {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(gate_up_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(2 * intermediate_dim), static_cast<size_t>(hidden_dim)}, dtype, device }));
//     INFINICORE_NN_PARAMETER_INIT(down_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim), static_cast<size_t>(intermediate_dim)}, dtype, device }));
// }

// GuMoeExperts::~GuMoeExperts() { 
//     if (handle_) infiniopDestroyHandle(handle_); 
// }

// Tensor GuMoeExperts::forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const {
//     if (hidden_states->dtype() != DataType::F32) throw std::runtime_error("F32 only");
    
//     Device device = hidden_states->device();
//     cudaStream_t stream = 0; 

//     size_t num_tokens = hidden_states->numel() / hidden_dim_;
//     int top_k = top_k_index->shape()[1];
//     size_t expanded_size = num_tokens * top_k;

//     auto print_shape = [](const Shape& s) {
//         std::string out = "[";
//         for(size_t i=0; i<s.size(); ++i) {
//             out += std::to_string(s[i]) + (i == s.size()-1 ? "" : ", ");
//         }
//         out += "]";
//         return out;
//     };

//     auto monitor_alloc = [&](const std::string& name, const Shape& shp, size_t unit_size) {
//         size_t total_bytes = 1;
//         for (auto s : shp) total_bytes *= s;
//         total_bytes *= unit_size;
//         std::cout << "[MEM_CHECK] Allocating [" << name << "]: Shape=" << print_shape(shp) 
//                   << ", MB=" << (total_bytes / (1024.0 * 1024.0)) << std::endl;
//     };

//     // --- Workspace 分配 ---
//     monitor_alloc("expert_counts", {(size_t)num_experts_}, sizeof(int32_t));
//     Tensor expert_counts = Tensor::zeros({(size_t)num_experts_}, DataType::I32, device);

//     monitor_alloc("expert_offsets", {(size_t)num_experts_ + 1}, sizeof(int32_t));
//     Tensor expert_offsets = Tensor::zeros({(size_t)num_experts_ + 1}, DataType::I32, device);

//     monitor_alloc("sorted_input", {expanded_size, (size_t)hidden_dim_}, sizeof(float));
//     Tensor sorted_input = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);

//     monitor_alloc("sorted_output", {expanded_size, (size_t)hidden_dim_}, sizeof(float));
//     Tensor sorted_output = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
    
//     monitor_alloc("sorted_row_map", {expanded_size}, sizeof(int32_t));
//     Tensor sorted_row_map = Tensor::empty({expanded_size}, DataType::I32, device);

//     monitor_alloc("sorted_weights", {expanded_size}, sizeof(float));
//     Tensor sorted_weights = Tensor::empty({expanded_size}, DataType::F32, device);
    
//     monitor_alloc("final_output", hidden_states->shape(), sizeof(float));
//     Tensor final_output = Tensor::zeros(hidden_states->shape(), DataType::F32, device);

//     float* d_input     = (float*)hidden_states->data();
//     int32_t* d_indices = (int32_t*)top_k_index->data();
//     float* d_values    = (float*)top_k_values->data();
//     int32_t* d_counts  = (int32_t*)expert_counts->data();
//     int32_t* d_offsets = (int32_t*)expert_offsets->data();

//     // Phase 1: 排序与重排 (增加检查点)
//     std::cout << "[CHECKPOINT] Launching moe_sort..." << std::endl;
//     launch_moe_sort(d_indices, d_counts, d_offsets, num_tokens, top_k, num_experts_, stream);
    
//     std::cout << "[CHECKPOINT] Launching moe_permute..." << std::endl;
//     launch_moe_permute(
//         d_input, d_indices, d_values, d_offsets,
//         (float*)sorted_input->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(),
//         d_counts, num_tokens, top_k, hidden_dim_, num_experts_, stream
//     );

//     // Phase 2: 计算
//     std::vector<int32_t> h_offsets(num_experts_ + 1);
//     std::cout << "[CHECKPOINT] Copying offsets to host..." << std::endl;
//     // 使用同步拷贝确保安全性
//     cudaMemcpy(h_offsets.data(), d_offsets, sizeof(int32_t) * (num_experts_ + 1), cudaMemcpyDeviceToHost);

//     for (int e = 0; e < num_experts_; ++e) {
//         int start_idx = h_offsets[e];
//         int count = h_offsets[e+1] - start_idx;
        
//         // 增加数据完整性校验，防止由于 Kernel 错误导致的非法内存申请
//         if (count < 0 || count > (int)expanded_size) {
//             std::cerr << "[FATAL] Expert " << e << " has invalid token count: " << count << std::endl;
//             continue;
//         }
//         if (count == 0) continue;

//         if (e % 20 == 0) std::cout << "[CHECKPOINT] Expert loop at " << e << ", count=" << count << std::endl;

//         Tensor expert_in = sorted_input->narrow({{0, (size_t)start_idx, (size_t)count}});
//         Tensor w_gate_up = gate_up_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)(2*intermediate_dim_), (size_t)hidden_dim_});
//         Tensor w_down = down_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)hidden_dim_, (size_t)intermediate_dim_});

//         Tensor gate_up_out = infinicore::op::linear(expert_in, w_gate_up, std::nullopt);
//         Tensor gate = gate_up_out->narrow({{1, 0, (size_t)intermediate_dim_}});
//         Tensor up = gate_up_out->narrow({{1, (size_t)intermediate_dim_, (size_t)intermediate_dim_}});
        
//         Tensor ffn_inner = infinicore::op::mul(infinicore::op::silu(gate), up, this->handle_);
//         Tensor expert_res = infinicore::op::linear(ffn_inner, w_down, std::nullopt);

//         float* dst_ptr = (float*)sorted_output->data() + start_idx * hidden_dim_;
//         cudaMemcpyAsync(dst_ptr, (float*)expert_res->data(), count * hidden_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//     }

//     // Phase 3: 还原
//     std::cout << "[CHECKPOINT] Launching moe_reduce..." << std::endl;
//     launch_moe_reduce(
//         (float*)sorted_output->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(),
//         (float*)final_output->data(), num_tokens, top_k, hidden_dim_, stream
//     );

//     return final_output;
// }

// // ==========================================
// // GuMoeSparseMoeBlock 实现
// // ==========================================

// GuMoeSparseMoeBlock::GuMoeSparseMoeBlock(int num_experts, int hidden_dim, int intermediate_dim, 
//                                          int top_k, bool norm_topk, 
//                                          const DataType& dtype, const Device& device) {
//     router_ = register_module<GuMoeTopkRounter>("router", num_experts, hidden_dim, top_k, norm_topk, dtype, device);
//     experts_ = register_module<GuMoeExperts>("experts", num_experts, hidden_dim, intermediate_dim, dtype, device);
// }

// Tensor GuMoeSparseMoeBlock::forward(const Tensor& hidden_states) {
//     auto input_shape = hidden_states->shape();
//     size_t batch_size = input_shape[0];
//     size_t seq_len = input_shape[1];
//     size_t hidden_dim = input_shape[2];
//     size_t total_tokens = hidden_states->numel() / hidden_dim;
//     Tensor hidden_states_reshaped = hidden_states->view({total_tokens, hidden_dim});
//     auto [routing_weights, selected_experts] = router_->forward(hidden_states_reshaped);
//     Tensor final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);
//     return final_hidden_states->view({batch_size, seq_len, hidden_dim});
// }

// } // namespace infinicore::nn

// #include "gu_moe.h" 

// #include <cstring>
// #include <stdexcept>
// #include <vector>
// #include <cstdint> 
// #include <tuple> 

// #include "src/nvidia_kernels/nvidia_kernels_moe.h"
// #include "infinicore/ops.hpp"
// #include "infinirt.h" 
// #include "infiniop.h" 
// #include "gu_mul.h"
// #include "gu_topk_softmax.h" 

// namespace infinicore::nn {

// // ==========================================
// // GuMoeTopkRounter 实现
// // ==========================================

// GuMoeTopkRounter::GuMoeTopkRounter(int num_experts, int hidden_dim, int top_k, bool norm_topk_prob, const DataType &dtype, const Device &device)
//     : top_k_(top_k), 
//       num_experts_(num_experts), 
//       hidden_dim_(hidden_dim), 
//       norm_topk_prob_(norm_topk_prob)
// {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(weight, ({ {static_cast<size_t>(num_experts_), static_cast<size_t>(hidden_dim_)}, dtype, device }));
// }

// GuMoeTopkRounter::~GuMoeTopkRounter() { 
//     if (handle_) infiniopDestroyHandle(handle_); 
// }

// std::pair<Tensor, Tensor> GuMoeTopkRounter::forward(const Tensor &hidden_states) const {
//     size_t total_tokens = hidden_states->numel() / hidden_dim_;
//     Tensor flattened = hidden_states->view({total_tokens, static_cast<size_t>(hidden_dim_)});
//     Tensor logits = infinicore::op::linear(flattened, weight_, std::nullopt);
//     auto [val, idx] = infinicore::op::topk_softmax(logits, top_k_, norm_topk_prob_, this->handle_);
//     return {val, idx};
// }

// // ==========================================
// // GuMoeExperts 实现
// // ==========================================

// GuMoeExperts::GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, const DataType& dtype, const Device& device)
//     : num_experts_(num_experts), 
//       hidden_dim_(hidden_dim), 
//       intermediate_dim_(intermediate_dim), 
//       device_(device) 
// {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(gate_up_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(2 * intermediate_dim), static_cast<size_t>(hidden_dim)}, dtype, device }));
//     INFINICORE_NN_PARAMETER_INIT(down_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim), static_cast<size_t>(intermediate_dim)}, dtype, device }));
// }

// GuMoeExperts::~GuMoeExperts() { 
//     if (handle_) infiniopDestroyHandle(handle_); 
// }

// Tensor GuMoeExperts::forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const {
//     if (hidden_states->dtype() != DataType::F32) throw std::runtime_error("F32 only");
    
//     Device device = hidden_states->device();
//     cudaStream_t stream = 0; 

//     size_t num_tokens = hidden_states->numel() / hidden_dim_;
//     int top_k = top_k_index->shape()[1];
//     size_t expanded_size = num_tokens * top_k;

//     // 1. 分配 Workspace (这些是持久的)
//     Tensor expert_counts = Tensor::zeros({(size_t)num_experts_}, DataType::I32, device);
//     Tensor expert_offsets = Tensor::zeros({(size_t)num_experts_ + 1}, DataType::I32, device);
//     Tensor sorted_input = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_output = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_row_map = Tensor::empty({expanded_size}, DataType::I32, device);
//     Tensor sorted_weights = Tensor::empty({expanded_size}, DataType::F32, device);
//     Tensor final_output = Tensor::zeros(hidden_states->shape(), DataType::F32, device);

//     float* d_input     = (float*)hidden_states->data();
//     int32_t* d_indices = (int32_t*)top_k_index->data();
//     float* d_values    = (float*)top_k_values->data();
//     int32_t* d_counts  = (int32_t*)expert_counts->data();
//     int32_t* d_offsets = (int32_t*)expert_offsets->data();

//     launch_moe_sort(d_indices, d_counts, d_offsets, num_tokens, top_k, num_experts_, stream);
//     launch_moe_permute(
//         d_input, d_indices, d_values, d_offsets,
//         (float*)sorted_input->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(),
//         d_counts, num_tokens, top_k, hidden_dim_, num_experts_, stream
//     );

//     // 2. 拷贝 Offset 必须同步，否则后面循环会乱
//     std::vector<int32_t> h_offsets(num_experts_ + 1);
//     cudaMemcpy(h_offsets.data(), d_offsets, sizeof(int32_t) * (num_experts_ + 1), cudaMemcpyDeviceToHost);

//     // 3. 专家循环：使用大括号控制局部变量生命周期
//     for (int e = 0; e < num_experts_; ++e) {
//         int start_idx = h_offsets[e];
//         int count = h_offsets[e+1] - start_idx;
//         if (count <= 0) continue;

//         {
//             // 在这个大括号内定义的 Tensor 会在每一轮迭代结束时立即析构
//             // 这能强制让 cudaMallocAsync 知道这块内存可以回收了
//             Tensor expert_in = sorted_input->narrow({{0, (size_t)start_idx, (size_t)count}});
//             Tensor w_gate_up = gate_up_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)(2*intermediate_dim_), (size_t)hidden_dim_});
//             Tensor w_down = down_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)hidden_dim_, (size_t)intermediate_dim_});

//             Tensor gate_up_out = infinicore::op::linear(expert_in, w_gate_up, std::nullopt);
//             Tensor gate = gate_up_out->narrow({{1, 0, (size_t)intermediate_dim_}});
//             Tensor up = gate_up_out->narrow({{1, (size_t)intermediate_dim_, (size_t)intermediate_dim_}});
            
//             Tensor ffn_inner = infinicore::op::mul(infinicore::op::silu(gate), up, this->handle_);
//             Tensor expert_res = infinicore::op::linear(ffn_inner, w_down, std::nullopt);

//             float* dst_ptr = (float*)sorted_output->data() + start_idx * hidden_dim_;
//             cudaMemcpyAsync(dst_ptr, (float*)expert_res->data(), count * hidden_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//         } // <--- 关键：在这里，上一轮的所有中间 Tensor 都会被释放
//     }

//     launch_moe_reduce(
//         (float*)sorted_output->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(),
//         (float*)final_output->data(), num_tokens, top_k, hidden_dim_, stream
//     );

//     // 4. 最终同步：解决全零问题的关键
//     cudaStreamSynchronize(stream);

//     return final_output;
// }

// // ==========================================
// // GuMoeSparseMoeBlock 实现
// // ==========================================

// GuMoeSparseMoeBlock::GuMoeSparseMoeBlock(int num_experts, int hidden_dim, int intermediate_dim, 
//                                          int top_k, bool norm_topk, 
//                                          const DataType& dtype, const Device& device) {
//     router_ = register_module<GuMoeTopkRounter>("router", num_experts, hidden_dim, top_k, norm_topk, dtype, device);
//     experts_ = register_module<GuMoeExperts>("experts", num_experts, hidden_dim, intermediate_dim, dtype, device);
// }

// Tensor GuMoeSparseMoeBlock::forward(const Tensor& hidden_states) {
//     auto input_shape = hidden_states->shape();
//     size_t batch_size = input_shape[0];
//     size_t seq_len = input_shape[1];
//     size_t hidden_dim = input_shape[2];
//     size_t total_tokens = hidden_states->numel() / hidden_dim;
//     Tensor hidden_states_reshaped = hidden_states->view({total_tokens, hidden_dim});
//     auto [routing_weights, selected_experts] = router_->forward(hidden_states_reshaped);
//     Tensor final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);
//     return final_hidden_states->view({batch_size, seq_len, hidden_dim});
// }

//} // namespace infinicore::nn

// #include "gu_moe.h" 
// #include <cstring>
// #include <stdexcept>
// #include <vector>
// #include <cstdint> 
// #include <tuple> 

// #include "src/nvidia_kernels/nvidia_kernels_moe.h"
// #include "infinicore/ops.hpp"
// #include "infinirt.h" 
// #include "infiniop.h" 
// #include "gu_mul.h"
// #include "gu_topk_softmax.h" 

// // 尝试引入框架的流获取接口
// namespace infinicore::context {
//     extern void* getStream();
// }

// namespace infinicore::nn {

// // ==========================================
// // GuMoeTopkRounter
// // ==========================================
// GuMoeTopkRounter::GuMoeTopkRounter(int num_experts, int hidden_dim, int top_k, bool norm_topk_prob, const DataType &dtype, const Device &device)
//     : top_k_(top_k), num_experts_(num_experts), hidden_dim_(hidden_dim), norm_topk_prob_(norm_topk_prob) {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(weight, ({ {static_cast<size_t>(num_experts_), static_cast<size_t>(hidden_dim_)}, dtype, device }));
// }

// GuMoeTopkRounter::~GuMoeTopkRounter() { if (handle_) infiniopDestroyHandle(handle_); }

// std::pair<Tensor, Tensor> GuMoeTopkRounter::forward(const Tensor &hidden_states) const {
//     size_t total_tokens = hidden_states->numel() / hidden_dim_;
//     Tensor flattened = hidden_states->view({total_tokens, static_cast<size_t>(hidden_dim_)});
//     Tensor logits = infinicore::op::linear(flattened, weight_, std::nullopt);
//     auto [val, idx] = infinicore::op::topk_softmax(logits, top_k_, norm_topk_prob_, this->handle_);
//     return {val, idx};
// }

// // ==========================================
// // GuMoeExperts
// // ==========================================
// GuMoeExperts::GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, const DataType& dtype, const Device& device)
//     : num_experts_(num_experts), hidden_dim_(hidden_dim), intermediate_dim_(intermediate_dim), device_(device) {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(gate_up_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(2 * intermediate_dim), static_cast<size_t>(hidden_dim)}, dtype, device }));
//     INFINICORE_NN_PARAMETER_INIT(down_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim), static_cast<size_t>(intermediate_dim)}, dtype, device }));
// }

// GuMoeExperts::~GuMoeExperts() { if (handle_) infiniopDestroyHandle(handle_); }

// Tensor GuMoeExperts::forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const {
//     Device device = hidden_states->device();
//     // 使用框架流，如果没有则退回到默认流 0
//     void* raw_stream = infinicore::context::getStream();
//     cudaStream_t stream = (cudaStream_t)raw_stream; //? (cudaStream_t)raw_stream : (cudaStream_t)0;

//     size_t num_tokens = hidden_states->numel() / hidden_dim_;
//     int top_k = top_k_index->shape()[1];
//     size_t expanded_size = num_tokens * top_k;

//     // 分配 Workspace
//     Tensor expert_counts = Tensor::zeros({(size_t)num_experts_}, DataType::I32, device);
//     Tensor expert_offsets = Tensor::zeros({(size_t)num_experts_ + 1}, DataType::I32, device);
//     Tensor sorted_input = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_output = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_row_map = Tensor::empty({expanded_size}, DataType::I32, device);
//     Tensor sorted_weights = Tensor::empty({expanded_size}, DataType::F32, device);
//     Tensor final_output = Tensor::zeros(hidden_states->shape(), DataType::F32, device);

//     // Phase 1
//     launch_moe_sort((int32_t*)top_k_index->data(), (int32_t*)expert_counts->data(), (int32_t*)expert_offsets->data(), num_tokens, top_k, num_experts_, stream);
//     launch_moe_permute((float*)hidden_states->data(), (int32_t*)top_k_index->data(), (float*)top_k_values->data(), (int32_t*)expert_offsets->data(),
//                        (float*)sorted_input->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(),
//                        (int32_t*)expert_counts->data(), num_tokens, top_k, hidden_dim_, num_experts_, stream);

//     // Phase 2
//     std::vector<int32_t> h_offsets(num_experts_ + 1);
//     cudaMemcpy(h_offsets.data(), expert_offsets->data(), sizeof(int32_t) * (num_experts_ + 1), cudaMemcpyDeviceToHost);

//     for (int e = 0; e < num_experts_; ++e) {
//         int start_idx = h_offsets[e];
//         int count = h_offsets[e+1] - start_idx;
//         if (count <= 0) continue;

//         { // 局部作用域回收显存
//             Tensor expert_in = sorted_input->narrow({{0, (size_t)start_idx, (size_t)count}});
//             Tensor w_gate_up = gate_up_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)(2*intermediate_dim_), (size_t)hidden_dim_});
//             Tensor w_down = down_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)hidden_dim_, (size_t)intermediate_dim_});

//             Tensor gate_up_out = infinicore::op::linear(expert_in, w_gate_up, std::nullopt);
//             Tensor gate = gate_up_out->narrow({{1, 0, (size_t)intermediate_dim_}});
//             Tensor up = gate_up_out->narrow({{1, (size_t)intermediate_dim_, (size_t)intermediate_dim_}});
//             Tensor ffn_inner = infinicore::op::mul(infinicore::op::silu(gate), up, this->handle_);
//             Tensor expert_res = infinicore::op::linear(ffn_inner, w_down, std::nullopt);

//             cudaMemcpyAsync((float*)sorted_output->data() + start_idx * hidden_dim_, (float*)expert_res->data(), count * hidden_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//         }
//     }

//     // Phase 3
//     launch_moe_reduce((float*)sorted_output->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(), (float*)final_output->data(), num_tokens, top_k, hidden_dim_, stream);
    
//     cudaStreamSynchronize(stream);
//     return final_output;
// }

// // ==========================================
// // GuMoeSparseMoeBlock
// // ==========================================
// GuMoeSparseMoeBlock::GuMoeSparseMoeBlock(int num_experts, int hidden_dim, int intermediate_dim, int top_k, bool norm_topk, const DataType& dtype, const Device& device) {
//     router_ = register_module<GuMoeTopkRounter>("router", num_experts, hidden_dim, top_k, norm_topk, dtype, device);
//     experts_ = register_module<GuMoeExperts>("experts", num_experts, hidden_dim, intermediate_dim, dtype, device);
// }

// Tensor GuMoeSparseMoeBlock::forward(const Tensor& hidden_states) {
//     size_t total_tokens = hidden_states->numel() / (hidden_states->shape().back());
//     Tensor hidden_states_reshaped = hidden_states->view({total_tokens, hidden_states->shape().back()});
//     auto [routing_weights, selected_experts] = router_->forward(hidden_states_reshaped);
//     Tensor final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);
//     return final_hidden_states->view(hidden_states->shape());
// }

// } // namespace infinicore::nn

// #include "gu_moe.h" 
// #include <cstring>
// #include <stdexcept>
// #include <vector>
// #include <cstdint> 
// #include <tuple> 
// #include <iostream>

// #include "src/nvidia_kernels/nvidia_kernels_moe.h"
// #include "infinicore/ops.hpp"
// #include "infinirt.h" 
// #include "infiniop.h" 
// #include "gu_mul.h"
// #include "gu_topk_softmax.h" 

// namespace infinicore::nn {

// // ==========================================
// // GuMoeTopkRounter
// // ==========================================
// GuMoeTopkRounter::GuMoeTopkRounter(int num_experts, int hidden_dim, int top_k, bool norm_topk_prob, const DataType &dtype, const Device &device)
//     : top_k_(top_k), num_experts_(num_experts), hidden_dim_(hidden_dim), norm_topk_prob_(norm_topk_prob) {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(weight, ({ {static_cast<size_t>(num_experts_), static_cast<size_t>(hidden_dim_)}, dtype, device }));
// }

// GuMoeTopkRounter::~GuMoeTopkRounter() { if (handle_) infiniopDestroyHandle(handle_); }

// std::pair<Tensor, Tensor> GuMoeTopkRounter::forward(const Tensor &hidden_states) const {
//     size_t total_tokens = hidden_states->numel() / hidden_dim_;
//     Tensor flattened = hidden_states->view({total_tokens, static_cast<size_t>(hidden_dim_)});
//     Tensor logits = infinicore::op::linear(flattened, weight_, std::nullopt);
//     auto [val, idx] = infinicore::op::topk_softmax(logits, top_k_, norm_topk_prob_, this->handle_);
//     return {val, idx};
// }

// // ==========================================
// // GuMoeExperts
// // ==========================================
// GuMoeExperts::GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, const DataType& dtype, const Device& device)
//     : num_experts_(num_experts), 
//       hidden_dim_(hidden_dim), 
//       intermediate_dim_(intermediate_dim), 
//       device_(device) 
// {
//     // --- 增加这一段强力打印 ---
//     printf("\n[CONSTRUCTOR_DEBUG] num_experts: %d, hidden: %d, inter: %d\n", 
//            num_experts, hidden_dim, intermediate_dim);
//     fflush(stdout); 

//     if (num_experts <= 0 || hidden_dim <= 0 || intermediate_dim <= 0) {
//         printf("[FATAL] Invalid dimensions detected!\n");
//         fflush(stdout);
//     }
//     // -------------------------

//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);

//     INFINICORE_NN_PARAMETER_INIT(gate_up_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(2 * intermediate_dim), static_cast<size_t>(hidden_dim)}, dtype, device }));
//     INFINICORE_NN_PARAMETER_INIT(down_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim), static_cast<size_t>(intermediate_dim)}, dtype, device }));
// }

// GuMoeExperts::~GuMoeExperts() { if (handle_) infiniopDestroyHandle(handle_); }

// Tensor GuMoeExperts::forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const {
//     Device device = hidden_states->device();
//     cudaStream_t stream = 0; 

//     size_t num_tokens = hidden_states->numel() / hidden_dim_;
//     int top_k = top_k_index->shape()[1];
//     size_t expanded_size = (size_t)num_tokens * top_k;

//     // 1. 显式分配 Workspace
//     Tensor expert_counts = Tensor::zeros({(size_t)num_experts_}, DataType::I32, device);
//     Tensor expert_offsets = Tensor::zeros({(size_t)num_experts_ + 1}, DataType::I32, device);
//     Tensor sorted_input = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_output = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_row_map = Tensor::empty({expanded_size}, DataType::I32, device);
//     Tensor sorted_weights = Tensor::empty({expanded_size}, DataType::F32, device);
//     Tensor final_output = Tensor::zeros(hidden_states->shape(), DataType::F32, device);

//     // Phase 1: 数据重排
//     launch_moe_sort((int32_t*)top_k_index->data(), (int32_t*)expert_counts->data(), (int32_t*)expert_offsets->data(), (int)num_tokens, top_k, num_experts_, stream);
//     launch_moe_permute((float*)hidden_states->data(), (int32_t*)top_k_index->data(), (float*)top_k_values->data(), (int32_t*)expert_offsets->data(),
//                        (float*)sorted_input->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(),
//                        (int32_t*)expert_counts->data(), (int)num_tokens, top_k, hidden_dim_, num_experts_, stream);

//     // Phase 2: 计算循环
//     std::vector<int32_t> h_offsets(num_experts_ + 1);
//     cudaMemcpy(h_offsets.data(), expert_offsets->data(), sizeof(int32_t) * (num_experts_ + 1), cudaMemcpyDeviceToHost);

//     for (int e = 0; e < num_experts_; ++e) {
//         int start_idx = h_offsets[e];
//         int count = h_offsets[e+1] - start_idx;
//         if (count <= 0) continue;

//         { // 利用作用域自动析构临时 Tensor，释放显存池
//             Tensor expert_in = sorted_input->narrow({{0, (size_t)start_idx, (size_t)count}});
//             Tensor w_gate_up = gate_up_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)(2*intermediate_dim_), (size_t)hidden_dim_});
//             Tensor w_down = down_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)hidden_dim_, (size_t)intermediate_dim_});

//             // 执行 FFN
//             Tensor gate_up_out = infinicore::op::linear(expert_in, w_gate_up, std::nullopt);
//             Tensor gate = gate_up_out->narrow({{1, 0, (size_t)intermediate_dim_}});
//             Tensor up = gate_up_out->narrow({{1, (size_t)intermediate_dim_, (size_t)intermediate_dim_}});
            
//             Tensor activated_gate = infinicore::op::silu(gate);
//             Tensor ffn_inner = infinicore::op::mul(activated_gate, up, this->handle_);
//             Tensor expert_res = infinicore::op::linear(ffn_inner, w_down, std::nullopt);

//             cudaMemcpyAsync((float*)sorted_output->data() + start_idx * hidden_dim_, (float*)expert_res->data(), (size_t)count * hidden_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//         } // 此处局部 Tensor 自动析构
//     }

//     // Phase 3: 结果规约
//     launch_moe_reduce((float*)sorted_output->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(), (float*)final_output->data(), (int)num_tokens, top_k, hidden_dim_, stream);
    
//     cudaStreamSynchronize(stream);
//     return final_output;
// }

// // ==========================================
// // GuMoeSparseMoeBlock
// // ==========================================
// GuMoeSparseMoeBlock::GuMoeSparseMoeBlock(int num_experts, int hidden_dim, int intermediate_dim, int top_k, bool norm_topk, const DataType& dtype, const Device& device) {
//     router_ = register_module<GuMoeTopkRounter>("router", num_experts, hidden_dim, top_k, norm_topk, dtype, device);
//     experts_ = register_module<GuMoeExperts>("experts", num_experts, hidden_dim, intermediate_dim, dtype, device);
// }

// Tensor GuMoeSparseMoeBlock::forward(const Tensor& hidden_states) {
//     auto shp = hidden_states->shape();
//     size_t last_dim = shp.back();
//     size_t total_tokens = hidden_states->numel() / last_dim;
    
//     Tensor hidden_states_reshaped = hidden_states->view({total_tokens, last_dim});
//     auto [routing_weights, selected_experts] = router_->forward(hidden_states_reshaped);
//     Tensor final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);
    
//     return final_hidden_states->view(shp);
// }

// } // namespace infinicore::nn

// #include "gu_moe.h" 
// #include <cstring>
// #include <stdexcept>
// #include <vector>
// #include <cstdint> 
// #include <tuple> 
// #include <iostream>

// #include "src/nvidia_kernels/nvidia_kernels_moe.h"
// #include "infinicore/ops.hpp"
// #include "infinirt.h" 
// #include "infiniop.h" 
// #include "gu_mul.h"
// #include "gu_topk_softmax.h" 

// // 引入框架流接口
// namespace infinicore::context {
//     extern void* getStream();
// }

// namespace infinicore::nn {

// // GuMoeTopkRounter (保持不变)
// GuMoeTopkRounter::GuMoeTopkRounter(int num_experts, int hidden_dim, int top_k, bool norm_topk_prob, const DataType &dtype, const Device &device)
//     : top_k_(top_k), num_experts_(num_experts), hidden_dim_(hidden_dim), norm_topk_prob_(norm_topk_prob) {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(weight, ({ {static_cast<size_t>(num_experts_), static_cast<size_t>(hidden_dim_)}, dtype, device }));
// }
// GuMoeTopkRounter::~GuMoeTopkRounter() { if (handle_) infiniopDestroyHandle(handle_); }
// std::pair<Tensor, Tensor> GuMoeTopkRounter::forward(const Tensor &hidden_states) const {
//     size_t total_tokens = hidden_states->numel() / hidden_dim_;
//     Tensor flattened = hidden_states->view({total_tokens, static_cast<size_t>(hidden_dim_)});
//     Tensor logits = infinicore::op::linear(flattened, weight_, std::nullopt);
//     auto [val, idx] = infinicore::op::topk_softmax(logits, top_k_, norm_topk_prob_, this->handle_);
//     return {val, idx};
// }

// // GuMoeExperts (保持不变)
// GuMoeExperts::GuMoeExperts(int num_experts, int hidden_dim, int intermediate_dim, const DataType& dtype, const Device& device)
//     : num_experts_(num_experts), hidden_dim_(hidden_dim), intermediate_dim_(intermediate_dim), device_(device) {
//     infinirtSetDevice((infiniDevice_t)device.getType(), device.getIndex());
//     infiniopCreateHandle(&this->handle_);
//     INFINICORE_NN_PARAMETER_INIT(gate_up_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(2 * intermediate_dim), static_cast<size_t>(hidden_dim)}, dtype, device }));
//     INFINICORE_NN_PARAMETER_INIT(down_proj, ({ {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim), static_cast<size_t>(intermediate_dim)}, dtype, device }));
// }
// GuMoeExperts::~GuMoeExperts() { if (handle_) infiniopDestroyHandle(handle_); }

// Tensor GuMoeExperts::forward(const Tensor& hidden_states, const Tensor& top_k_index, const Tensor& top_k_values) const {
//     Device device = hidden_states->device();
//     void* raw_stream = infinicore::context::getStream();
//     cudaStream_t stream = raw_stream ? (cudaStream_t)raw_stream : 0;

//     // 回退类型转换，直接使用原始指针，但在 count 计算处做防御
//     size_t num_tokens = hidden_states->numel() / hidden_dim_;
//     int top_k = top_k_index->shape()[1];
//     size_t expanded_size = num_tokens * top_k;

//     Tensor expert_counts = Tensor::zeros({(size_t)num_experts_}, DataType::I32, device);
//     Tensor expert_offsets = Tensor::zeros({(size_t)num_experts_ + 1}, DataType::I32, device);
//     Tensor sorted_input = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_output = Tensor::empty({expanded_size, (size_t)hidden_dim_}, DataType::F32, device);
//     Tensor sorted_row_map = Tensor::empty({expanded_size}, DataType::I32, device);
//     Tensor sorted_weights = Tensor::empty({expanded_size}, DataType::F32, device);
//     Tensor final_output = Tensor::zeros(hidden_states->shape(), DataType::F32, device);

//     launch_moe_sort(
//         (int32_t*)top_k_index->data(), 
//         (int32_t*)expert_counts->data(), 
//         (int32_t*)expert_offsets->data(), 
//         (int)num_tokens, top_k, num_experts_, stream
//     );
    
//     launch_moe_permute(
//         (float*)hidden_states->data(), 
//         (int32_t*)top_k_index->data(), 
//         (float*)top_k_values->data(), 
//         (int32_t*)expert_offsets->data(),
//         (float*)sorted_input->data(), 
//         (int32_t*)sorted_row_map->data(), 
//         (float*)sorted_weights->data(),
//         (int32_t*)expert_counts->data(), 
//         (int)num_tokens, top_k, hidden_dim_, num_experts_, stream
//     );

//     std::vector<int32_t> h_offsets(num_experts_ + 1);
//     cudaMemcpy(h_offsets.data(), expert_offsets->data(), sizeof(int32_t) * (num_experts_ + 1), cudaMemcpyDeviceToHost);

//     for (int e = 0; e < num_experts_; ++e) {
//         int start_idx = h_offsets[e];
//         int count = h_offsets[e+1] - start_idx;
        
//         // 【核心防御】防止 Error 700 / OOM
//         // 如果 count 异常（可能是由于 Int64/32 读取错位导致的），直接跳过！
//         if (count <= 0 || count > (int)expanded_size) {
//             if (count > (int)expanded_size) {
//                 printf("WARNING: Expert %d skipped due to invalid count: %d\n", e, count);
//             }
//             printf("WARNING: Expert %d skipped due to invalid count: %d\n", e, count);
//             continue;
//         }

//         { 
//             Tensor expert_in = sorted_input->narrow({{0, (size_t)start_idx, (size_t)count}});
//             Tensor w_gate_up = gate_up_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)(2*intermediate_dim_), (size_t)hidden_dim_});
//             Tensor w_down = down_proj_->narrow({{0, (size_t)e, 1}})->view({(size_t)hidden_dim_, (size_t)intermediate_dim_});

//             Tensor gate_up_out = infinicore::op::linear(expert_in, w_gate_up, std::nullopt);
//             Tensor gate = gate_up_out->narrow({{1, 0, (size_t)intermediate_dim_}});
//             Tensor up = gate_up_out->narrow({{1, (size_t)intermediate_dim_, (size_t)intermediate_dim_}});
            
//             Tensor ffn_inner = infinicore::op::mul(infinicore::op::silu(gate), up, this->handle_);
//             Tensor expert_res = infinicore::op::linear(ffn_inner, w_down, std::nullopt);

//             cudaMemcpyAsync((float*)sorted_output->data() + start_idx * hidden_dim_, (float*)expert_res->data(), (size_t)count * hidden_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//         }
//     }

//     launch_moe_reduce((float*)sorted_output->data(), (int32_t*)sorted_row_map->data(), (float*)sorted_weights->data(), (float*)final_output->data(), (int)num_tokens, top_k, hidden_dim_, stream);
    
//     cudaStreamSynchronize(stream);
//     return final_output;
// }

// // GuMoeSparseMoeBlock (保持不变)
// GuMoeSparseMoeBlock::GuMoeSparseMoeBlock(int num_experts, int hidden_dim, int intermediate_dim, int top_k, bool norm_topk, const DataType& dtype, const Device& device) {
//     router_ = register_module<GuMoeTopkRounter>("router", num_experts, hidden_dim, top_k, norm_topk, dtype, device);
//     experts_ = register_module<GuMoeExperts>("experts", num_experts, hidden_dim, intermediate_dim, dtype, device);
// }
// Tensor GuMoeSparseMoeBlock::forward(const Tensor& hidden_states) {
//     size_t total_tokens = hidden_states->numel() / (hidden_states->shape().back());
//     Tensor hidden_states_reshaped = hidden_states->view({total_tokens, hidden_states->shape().back()});
//     auto [routing_weights, selected_experts] = router_->forward(hidden_states_reshaped);
//     Tensor final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);
//     return final_hidden_states->view(hidden_states->shape());
// }

// } // namespace infinicore::nn

// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <cstdio>
// #include <cub/cub.cuh>

// #define MAX_EXPERTS 256

// #define CUDA_CHECK(call) \
// do { \
//     cudaError_t error = call; \
//     if (error != cudaSuccess) { \
//         fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(error)); \
//         exit(1); \
//     } \
// } while(0)

// __global__ void count_kernel_sota(
//     const int32_t* __restrict__ topk_ids, 
//     int32_t* __restrict__ expert_counts,  
//     int total_tasks,
//     int num_experts
// ) {
//     extern __shared__ int32_t smem_counts[]; 
    
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int gid = bid * blockDim.x + tid;

//     for (int i = tid; i < num_experts; i += blockDim.x) {
//         smem_counts[i] = 0;
//     }
//     __syncthreads();

//     if (gid < total_tasks) {
//         int expert_id = topk_ids[gid];

//         unsigned int active_mask = __activemask();
//         unsigned int mask = __match_any_sync(active_mask, expert_id);

//         int leader = __ffs(mask) - 1; // Find First Set
//         int lane_id = tid % 32;

//         if (lane_id == leader) {

//             int agg_count = __popc(mask);
            
//             atomicAdd(&smem_counts[expert_id], agg_count);
//         }
//     }
    
//     __syncthreads();

//     for (int i = tid; i < num_experts; i += blockDim.x) {
//         int count = smem_counts[i];
//         if (count > 0) {
//             atomicAdd(&expert_counts[i], count);
//         }
//     }
// }

// void launch_moe_sort(
//     const int32_t* topk_ids,
//     int32_t* expert_counts,   
//     int32_t* expert_offsets, // 长度建议是 num_experts + 1
//     int num_tokens,
//     int top_k,
//     int num_experts,
//     cudaStream_t stream
// ) {
//     int total_tasks = num_tokens * top_k;
//     int block_size = 256;
//     int grid_size = (total_tasks + block_size - 1) / block_size;

//     // -------------------------------------------------------
//     CUDA_CHECK(cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), stream));
    
//     count_kernel_sota<<<grid_size, block_size, num_experts * sizeof(int32_t), stream>>>(
//         topk_ids, expert_counts, total_tasks, num_experts
//     );
    
//     void* d_temp_storage = NULL;
//     size_t temp_storage_bytes = 0;
    
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
//                                   expert_counts,  // 输入: counts
//                                   expert_offsets, // 输出: offsets
//                                   num_experts + 1,// 长度: 多算一位作为总和
//                                   stream);
    
//     CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
    
//     // 执行
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
//                                   expert_counts, 
//                                   expert_offsets, 
//                                   num_experts + 1, 
//                                   stream);
                                  
//     CUDA_CHECK(cudaFreeAsync(d_temp_storage, stream));
// }

// __global__ void permute_kernel(
//     const float* __restrict__ input,           // [N, H] 源数据
//     const int32_t* __restrict__ topk_ids,      // [N, K] 路由
//     const float* __restrict__ topk_weights,
//     const int32_t* __restrict__ expert_offsets,// [E]    起始位置
//     int32_t* __restrict__ running_counters,    // [E]    临时计数器 (原子加专用)
//     float* __restrict__ sorted_input,          // [N*K, H] 目标数据
//     int32_t* __restrict__ sorted_row_map,      // [N*K]    来源记录
//     float* __restrict__ sorted_weights,
//     int num_tokens,
//     int top_k,
//     int hidden_dim
// ) {
//     // 任务总数 = Token数 * TopK (因为可能有复制)
//     int total_tasks = num_tokens * top_k;
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (tid >= total_tasks) return;

//     int token_idx = tid / top_k;
//     // int k_idx = tid % top_k; // 如果 weights 是 [N, K] 布局，需要用这个
//     int expert_id = topk_ids[tid];

//     // 获取该专家的起始地址
//     int base_offset = expert_offsets[expert_id];
//     // 原子获取我是该专家的第几个客人
//     int my_rank = atomicAdd(&running_counters[expert_id], 1);
//     // 计算最终写入的行号
//     int target_row = base_offset + my_rank;

//     // 记下：第 target_row 行数据，其实是原来的 token_idx
//     sorted_row_map[target_row] = token_idx;
//     sorted_weights[target_row] = topk_weights[tid];

//     // 从 input[token_idx] 搬到 sorted_input[target_row]
//     const float* src_ptr = input + token_idx * hidden_dim;
//     float* dst_ptr = sorted_input + target_row * hidden_dim;

//     // 尝试使用 float4 (128-bit) 进行搬运，减少指令数
//     int vec_size = hidden_dim / 4;
//     int remainder = hidden_dim % 4;
    
//     // 强转指针进行向量化读取
//     const float4* src_vec = (const float4*)src_ptr;
//     float4* dst_vec = (float4*)dst_ptr;

//     for (int i = 0; i < vec_size; ++i) {
//         dst_vec[i] = src_vec[i];
//     }
//     // 处理剩下的尾巴 (如果有的话)
//     for (int i = 0; i < remainder; ++i) {
//         int idx = vec_size * 4 + i;
//         dst_ptr[idx] = src_ptr[idx];
//     }
// }

// void launch_moe_permute(
//     const float* input,
//     const int32_t* topk_ids,
//     const float* topk_weights,
//     const int32_t* expert_offsets,
//     float* sorted_input,
//     int32_t* sorted_row_map,
//     float* sorted_weights,
//     int32_t* expert_counts, // <--- 复用这个数组作为临时计数器
//     int num_tokens,
//     int top_k,
//     int hidden_dim,
//     int num_experts,
//     cudaStream_t stream
// ) {
//     int total_tasks = num_tokens * top_k;
//     int block_size = 256;
//     int grid_size = (total_tasks + block_size - 1) / block_size;

//     // 1. 【关键】把计数器重置为 0
//     // 这样每个专家才能从第 0 个开始数
//     CUDA_CHECK(cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), stream));

//     // 2. 启动 Kernel
//     permute_kernel<<<grid_size, block_size, 0, stream>>>(
//         input, 
//         topk_ids, 
//         topk_weights,
//         expert_offsets, 
//         expert_counts, // 这里传进去当作 running_counters 用
//         sorted_input, 
//         sorted_row_map,
//         sorted_weights,
//         num_tokens, 
//         top_k, 
//         hidden_dim
//     );
// }

// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <cstdio>
// #include <cub/cub.cuh>

// #define MAX_EXPERTS 256

// // 增强版 Check 宏
// #define CUDA_CHECK(call) \
// do { \
//     cudaError_t error = call; \
//     if (error != cudaSuccess) { \
//         fprintf(stderr, "[KERNEL ERROR] %s failed at line %d: %s\n", #call, __LINE__, cudaGetErrorString(error)); \
//         exit(1); \
//     } \
// } while(0)

// // =============================================================
// // 1. Count Kernel (统计每个专家的 token 数)
// // =============================================================
// __global__ void count_kernel_sota(
//     const int32_t* __restrict__ topk_ids, 
//     int32_t* __restrict__ expert_counts,  
//     int total_tasks,
//     int num_experts
// ) {
//     extern __shared__ int32_t smem_counts[]; 
    
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int gid = bid * blockDim.x + tid;

//     // 初始化共享内存
//     for (int i = tid; i < num_experts; i += blockDim.x) {
//         smem_counts[i] = 0;
//     }
//     __syncthreads();

//     // 统计
//     if (gid < total_tasks) {
//         int expert_id = topk_ids[gid];
//         // 简单的边界检查
//         if (expert_id >= 0 && expert_id < num_experts) {
//             unsigned int active_mask = __activemask();
//             unsigned int mask = __match_any_sync(active_mask, expert_id);
//             int leader = __ffs(mask) - 1; 
//             int lane_id = tid % 32;
//             if (lane_id == leader) {
//                 int agg_count = __popc(mask);
//                 atomicAdd(&smem_counts[expert_id], agg_count);
//             }
//         }
//     }
//     __syncthreads();

//     // 写回全局内存
//     for (int i = tid; i < num_experts; i += blockDim.x) {
//         int count = smem_counts[i];
//         if (count > 0) {
//             atomicAdd(&expert_counts[i], count);
//         }
//     }
// }

// void launch_moe_sort(
//     const int32_t* topk_ids,
//     int32_t* expert_counts,   
//     int32_t* expert_offsets, 
//     int num_tokens,
//     int top_k,
//     int num_experts,
//     cudaStream_t stream
// ) {
//     int total_tasks = num_tokens * top_k;
//     int block_size = 256;
//     int grid_size = (total_tasks + block_size - 1) / block_size;

//     // 清零 Counts
//     CUDA_CHECK(cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), stream));
    
//     // 运行统计
//     count_kernel_sota<<<grid_size, block_size, num_experts * sizeof(int32_t), stream>>>(
//         topk_ids, expert_counts, total_tasks, num_experts
//     );
    
//     // CUB Scan (前缀和)
//     void* d_temp_storage = NULL;
//     size_t temp_storage_bytes = 0;
    
//     // 查询所需显存 (注意 num_experts + 1 以计算总和)
//     // 这里的 expert_counts 对应 gumoe.cpp 里申请的 (num_experts + 1) 大小，安全。
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
//                                   expert_counts, expert_offsets, 
//                                   num_experts + 1, stream);
    
//     // ====================================================
//     // 【关键修改】使用同步 cudaMalloc
//     // 必须替换掉原来的 cudaMallocAsync，否则在你的环境里会分配失败导致 Core Dump
//     // ====================================================
//     CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
//     // 执行 Scan
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
//                                   expert_counts, expert_offsets, 
//                                   num_experts + 1, stream);
                                  
//     // 同步释放
//     CUDA_CHECK(cudaFree(d_temp_storage));
// }

// // =============================================================
// // 2. Permute Kernel (重排数据)
// // =============================================================
// __global__ void permute_kernel(
//     const float* __restrict__ input,           
//     const int32_t* __restrict__ topk_ids,      
//     const float* __restrict__ topk_weights,
//     const int32_t* __restrict__ expert_offsets,
//     int32_t* __restrict__ running_counters,    
//     float* __restrict__ sorted_input,          
//     int32_t* __restrict__ sorted_row_map,      
//     float* __restrict__ sorted_weights,
//     int num_tokens,
//     int top_k,
//     int hidden_dim
// ) {
//     int total_tasks = num_tokens * top_k;
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (tid >= total_tasks) return;

//     int token_idx = tid / top_k;
//     int expert_id = topk_ids[tid];

//     // 原子获取写入位置
//     int base_offset = expert_offsets[expert_id];
//     int my_rank = atomicAdd(&running_counters[expert_id], 1);
//     int target_row = base_offset + my_rank;

//     // 记录映射关系
//     if (sorted_row_map) sorted_row_map[target_row] = token_idx;
//     if (sorted_weights) sorted_weights[target_row] = topk_weights[tid];

//     // 搬运 Hidden States
//     const float* src_ptr = input + token_idx * hidden_dim;
//     float* dst_ptr = sorted_input + target_row * hidden_dim;

//     // 简单的 float4 优化
//     int vec_size = hidden_dim / 4;
//     int remainder = hidden_dim % 4;
//     const float4* src_vec = (const float4*)src_ptr;
//     float4* dst_vec = (float4*)dst_ptr;

//     for (int i = 0; i < vec_size; ++i) {
//         dst_vec[i] = src_vec[i];
//     }
//     for (int i = 0; i < remainder; ++i) {
//         int idx = vec_size * 4 + i;
//         dst_ptr[idx] = src_ptr[idx];
//     }
// }

// void launch_moe_permute(
//     const float* input,
//     const int32_t* topk_ids,
//     const float* topk_weights,
//     const int32_t* expert_offsets,
//     float* sorted_input,
//     int32_t* sorted_row_map,
//     float* sorted_weights,
//     int32_t* expert_counts, 
//     int num_tokens,
//     int top_k,
//     int hidden_dim,
//     int num_experts,
//     cudaStream_t stream
// ) {
//     int total_tasks = num_tokens * top_k;
//     int block_size = 256;
//     int grid_size = (total_tasks + block_size - 1) / block_size;

//     // 复用 expert_counts 作为计数器，必须清零
//     CUDA_CHECK(cudaMemsetAsync(expert_counts, 0, (num_experts + 1)* sizeof(int32_t), stream));

//     permute_kernel<<<grid_size, block_size, 0, stream>>>(
//         input, topk_ids, topk_weights, expert_offsets, expert_counts, 
//         sorted_input, sorted_row_map, sorted_weights,
//         num_tokens, top_k, hidden_dim
//     );
// }

// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <cstdio>
// #include <cub/cub.cuh>
// #include <vector>
// #define MAX_EXPERTS 256

// // 错误检查宏
// #define CUDA_CHECK(call) \
// do { \
//     cudaError_t error = call; \
//     if (error != cudaSuccess) { \
//         fprintf(stderr, "[KERNEL ERROR] %s failed at line %d: %s\n", #call, __LINE__, cudaGetErrorString(error)); \
//         exit(1); \
//     } \
// } while(0)

// // ==========================================================
// // 【新武器】GPU 数据探针
// // ==========================================================
// __global__ void debug_inspector(int32_t* counts, int32_t* offsets, int num_experts) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         printf("\n[GPU INSPECTOR] --- Start Analysis ---\n");
        
//         // 1. 检查 Counts (前10个)
//         printf("[GPU] Counts (First 10): ");
//         bool counts_all_zero = true;
//         for(int i=0; i<min(10, num_experts); ++i) {
//             printf("%d ", counts[i]);
//             if (counts[i] != 0) counts_all_zero = false;
//         }
//         printf("\n");

//         // 2. 检查 Offsets (前10个 和 最后一个)
//         printf("[GPU] Offsets (First 10): ");
//         for(int i=0; i<min(10, num_experts); ++i) printf("%d ", offsets[i]);
//         printf("... Last(Total): %d\n", offsets[num_experts]);

//         // 3. 实时诊断
//         if (offsets[0] > 1000000000 || offsets[0] < 0) {
//             printf("[GPU CRITICAL] Offsets[0] is garbage! CUB Scan failed.\n");
//         }
//         if (counts_all_zero && offsets[num_experts] == 0) {
//             printf("[GPU WARNING] Counts are all zero. Input indices might be wrong.\n");
//         }
//         printf("[GPU INSPECTOR] --- End Analysis ---\n\n");
//     }
// }

// // ----------------------------------------------------------
// // Count Kernel (保持不变)
// // ----------------------------------------------------------
// __global__ void count_kernel_sota(
//     const int32_t* __restrict__ topk_ids, 
//     int32_t* __restrict__ expert_counts,  
//     int total_tasks,
//     int num_experts
// ) {
//     extern __shared__ int32_t smem_counts[]; 
//     int tid = threadIdx.x;
//     int gid = blockIdx.x * blockDim.x + tid;
//     if (gid == 0) {
//         printf("[GPU ALIVE] Kernel started. total_tasks=%d, num_experts=%d\n", total_tasks, num_experts);
//         printf("[GPU DATA] First topk_id = %d\n", topk_ids[0]); 
//     }
//     for (int i = tid; i < num_experts; i += blockDim.x) smem_counts[i] = 0;
//     __syncthreads();

//     if (gid < total_tasks) {
//         int expert_id = topk_ids[gid];
//         if (expert_id >= 0 && expert_id < num_experts) {
//             unsigned int mask = __match_any_sync(__activemask(), expert_id);
//             if ((tid % 32) == (__ffs(mask) - 1)) {
//                 atomicAdd(&smem_counts[expert_id], __popc(mask));
//             }
//         }
//     }
//     __syncthreads();
//     for (int i = tid; i < num_experts; i += blockDim.x) {
//         if (smem_counts[i] > 0) atomicAdd(&expert_counts[i], smem_counts[i]);
//         // printf("这是count_kernel_sota的数字%d\n", smem_counts[i]);
//     }
// }

// // ----------------------------------------------------------
// // Sort Launch (植入了探针)
// // ----------------------------------------------------------
// // void launch_moe_sort(
// //     const int32_t* topk_ids,
// //     int32_t* expert_counts,   
// //     int32_t* expert_offsets, 
// //     int num_tokens,
// //     int top_k,
// //     int num_experts,
// //     cudaStream_t stream
// // ) {
// //     int total_tasks = num_tokens * top_k;
// //     int block_size = 256;
// //     int grid_size = (total_tasks + block_size - 1) / block_size;
// //     printf("6\n");
// //     // 清零 (注意：这里用同步 memset 以排除异步干扰)
// //     CUDA_CHECK(cudaMemset(expert_counts, 0, (num_experts + 1) * sizeof(int32_t)));
    
// //     count_kernel_sota<<<grid_size, block_size, num_experts * sizeof(int32_t), stream>>>(
// //         topk_ids, expert_counts, total_tasks, num_experts
// //     );
// //     printf("7\n");
// //     // CUB Scan
// //     void* d_temp_storage = NULL;
// //     size_t temp_storage_bytes = 0;
    
// //     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
// //                                   expert_counts, expert_offsets, 
// //                                   num_experts + 1, stream);
// //     printf("8\n");
// //     // 【强制同步分配】确保 Scan 内存绝对可用
// //     CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
// //     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
// //                                   expert_counts, expert_offsets, 
// //                                   num_experts + 1, stream);
// //     printf("9\n");
// //     CUDA_CHECK(cudaFree(d_temp_storage));
// // }
// void launch_moe_sort(
//     const int32_t* topk_ids,
//     int32_t* expert_counts,   
//     int32_t* expert_offsets, 
//     int num_tokens,
//     int top_k,
//     int num_experts,
//     cudaStream_t stream
// ) {
//     int total_tasks = num_tokens * top_k;
//     int block_size = 256;
//     int grid_size = (total_tasks + block_size - 1) / block_size;

//     printf("6 - Preparing to launch count_kernel\n");
    
//     // 1. 清零
//     CUDA_CHECK(cudaMemsetAsync(expert_counts, 0, (num_experts + 1) * sizeof(int32_t), stream));
    
//     // 2. 计算共享内存大小 (关键！)
//     size_t smem_size = (num_experts + 1) * sizeof(int32_t);
    
//     // [DEBUG] 打印启动参数，看看是不是有 0
//     printf(">>> Launch Params: Grid=%d, Block=%d, SharedMem=%zu bytes, Experts=%d\n", 
//            grid_size, block_size, smem_size, num_experts);

//     // 3. 启动 Kernel
//     count_kernel_sota<<<grid_size, block_size, smem_size, stream>>>(
//         topk_ids, expert_counts, total_tasks, num_experts
//     );

//     // =========================================================
//     // 【捕获启动失败】这是你没看到 printf 的真正原因
//     // =========================================================
//     cudaError_t launch_err = cudaGetLastError();
//     if (launch_err != cudaSuccess) {
//         printf("❌ [FATAL] Kernel Launch Failed! Code=%d, Msg=%s\n", 
//                launch_err, cudaGetErrorString(launch_err));
//         // 这里不要 exit，打印出来让我们看到原因
//     } else {
//         printf("✅ Kernel Launch Requested Successfully.\n");
//     }

//     // 4. CUB Scan (保持你现在的代码)
//     void* d_temp_storage = NULL;
//     size_t temp_storage_bytes = 0;
    
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
//                                   expert_counts, expert_offsets, 
//                                   num_experts + 1, stream);
    
//     CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
//                                   expert_counts, expert_offsets, 
//                                   num_experts + 1, stream);
                                  
//     CUDA_CHECK(cudaFree(d_temp_storage));
// }

// // ----------------------------------------------------------
// // Permute Kernel (保持不变)
// // ----------------------------------------------------------
// __global__ void permute_kernel(
//     const float* __restrict__ input,           
//     const int32_t* __restrict__ topk_ids,      
//     const float* __restrict__ topk_weights,
//     const int32_t* __restrict__ expert_offsets,
//     int32_t* __restrict__ running_counters,    
//     float* __restrict__ sorted_input,          
//     int32_t* __restrict__ sorted_row_map,      
//     float* __restrict__ sorted_weights,
//     int num_tokens,
//     int top_k,
//     int hidden_dim
// ) {
//     int total_tasks = num_tokens * top_k;
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (tid >= total_tasks) return;

//     int token_idx = tid / top_k;
//     int expert_id = topk_ids[tid];

//     int base_offset = expert_offsets[expert_id];
//     int my_rank = atomicAdd(&running_counters[expert_id], 1);
//     int target_row = base_offset + my_rank;

//     if (sorted_row_map) sorted_row_map[target_row] = token_idx;
//     if (sorted_weights) sorted_weights[target_row] = topk_weights[tid];

//     const float* src_ptr = input + token_idx * hidden_dim;
//     float* dst_ptr = sorted_input + target_row * hidden_dim;

//     for (int i = 0; i < hidden_dim; ++i) dst_ptr[i] = src_ptr[i];
// }

// void launch_moe_permute(
//     const float* input,
//     const int32_t* topk_ids,
//     const float* topk_weights,
//     const int32_t* expert_offsets,
//     float* sorted_input,
//     int32_t* sorted_row_map,
//     float* sorted_weights,
//     int32_t* expert_counts, 
//     int num_tokens,
//     int top_k,
//     int hidden_dim,
//     int num_experts,
//     cudaStream_t stream
// ) {
//     int block_size = 256;
//     int grid_size = (num_tokens * top_k + block_size - 1) / block_size;

//     // 清零 running_counters
//     CUDA_CHECK(cudaMemset(expert_counts, 0, (num_experts + 1) * sizeof(int32_t)));

//     permute_kernel<<<grid_size, block_size, 0, stream>>>(
//         input, topk_ids, topk_weights, expert_offsets, expert_counts, 
//         sorted_input, sorted_row_map, sorted_weights,
//         num_tokens, top_k, hidden_dim
//     );
// }


// # import os
// # import torch
// # # 【强制黑魔法】告诉 PyTorch 我们要用新版 ABI
// # # 这行代码能救命，它防止 PyTorch 自己把 flag 改回 0
// # torch._C._GLIBCXX_USE_CXX11_ABI = True

// # from setuptools import setup
// # from torch.utils.cpp_extension import BuildExtension, CUDAExtension
// # import pybind11

// # INFINI_SRC_ROOT = "/data/users/shankgu/InfiniCore" 
// # INFINI_LM_ROOT = "/data/users/shankgu/InfiniLM"
// # INFINI_LIB_DIR = "/data/users/shankgu/InfiniCore/build/linux/x86_64/release"

// # # 你的库列表 (保持你之前的配置)
// # libs = [
// #     # 如果你的 gumoe.cpp 继承了 Module，你需要链接 utils 库
// #     os.path.join(INFINI_LIB_DIR, 'libinfini-utils.a'), 
// #     os.path.join(INFINI_LIB_DIR, 'libinfiniop-nvidia.a'),
// #     os.path.join(INFINI_LIB_DIR, 'libinfiniccl-nvidia.a'),
// #     os.path.join(INFINI_LIB_DIR, 'libinfinirt-nvidia.a') 
// # ]

// # setup(
// #     name='gu_moe_ops',
// #     version='0.1.0',
// #     ext_modules=[
// #         CUDAExtension(
// #             name='gu_moe_ops',
// #             sources=[
// #                 'pybind_gumoe.cc',          
// #                 'src/gumoe.cpp',            
// #                 'src/gu_mul.cc',            
// #                 'src/gu_topk_softmax.cc',
// #                 'src/nvidia_kernels/gu_reduce.cu',
// #                 'src/nvidia_kernels/gu_sort.cu',    
// #             ],
// #             include_dirs=[
// #                 pybind11.get_include(),
// #                 os.path.join(INFINI_SRC_ROOT, 'include'),
// #                 os.path.join(INFINI_LM_ROOT, 'src'),
// #                 'src'                       
// #             ],
// #             extra_objects=libs,
            
// #             extra_compile_args={
// #                 # 【唯一关键点】必须设为 1，解决 ...ESs 报错
// #                 'cxx': ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=1'],
// #                 'nvcc': ['-O3']
// #             }
// #         )
// #     ],
// #     cmdclass={
// #         'build_ext': BuildExtension
// #     }
// # )

// #include <cuda_runtime.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/numpy.h>
// #include <vector>
// #include <iostream>

// #include "src/gu_moe.h"
// #include "infinicore/tensor.hpp"
// #include "infinicore/device.hpp"

// namespace py = pybind11;
// using namespace infinicore;

// // int_to_dtype 略 (保持不变)...
// infinicore::DataType int_to_dtype(int id) {
//     switch (id) {
//         case 0: return infinicore::DataType::F32;
//         case 1: return infinicore::DataType::BF16;
//         case 2: return infinicore::DataType::I32;
//         case 3: return infinicore::DataType::F16;
//         default: throw std::runtime_error("Unknown dtype id: " + std::to_string(id));
//     }
// }

// class PyGuMoeWrapper {
// public:
//     std::shared_ptr<nn::GuMoeSparseMoeBlock> block;

//     PyGuMoeWrapper(int num_experts, int hidden_dim, int intermediate_dim, 
//                    int dtype_id, int device_id) {
//         Device device(Device::Type::NVIDIA, device_id);
//         block = std::make_shared<nn::GuMoeSparseMoeBlock>(
//             num_experts, hidden_dim, intermediate_dim, 2, true, 
//             int_to_dtype(dtype_id), device
//         );
//     }

//     infinicore::nn::Parameter object_to_tensor(py::object tensor_obj) {
//         uint64_t ptr_val = tensor_obj.attr("ptr").cast<uint64_t>();
//         void* raw_ptr = reinterpret_cast<void*>(ptr_val);
//         std::vector<int64_t> shape_vec = tensor_obj.attr("shape").cast<std::vector<int64_t>>();
//         infinicore::Shape shape;
//         for(auto s : shape_vec) shape.push_back(s);
//         int dtype_id = tensor_obj.attr("dtype_id").cast<int>();
//         int dev_id = tensor_obj.attr("device_id").cast<int>();
//         infinicore::Device dev(infinicore::Device::Type::NVIDIA, dev_id);
//         return infinicore::Tensor::from_blob(raw_ptr, shape, int_to_dtype(dtype_id), dev);
//     }

//     // ✅✅✅ 【关键】这里只有 2 个参数！
//     void forward(py::object input_obj, py::object output_obj) {
//         auto input = object_to_tensor(input_obj);
        
//         // 调用 C++ Block (单参数 forward)
//         auto internal_result = block->forward(input);
        
//         auto output_buffer = object_to_tensor(output_obj);
//         size_t bytes = internal_result->numel() * 4;
//         cudaMemcpy(output_buffer->data(), internal_result->data(), bytes, cudaMemcpyDeviceToDevice);
//     }

//     // ✅ set_weights 接收 3 个参数：GateUp, Down, RouterWeight
//     void set_weights(py::object gate_up_obj, py::object down_obj, py::object router_w_obj) {
//         auto gate_up = object_to_tensor(gate_up_obj);
//         auto down = object_to_tensor(down_obj);
//         auto router_w = object_to_tensor(router_w_obj);
        
//         block->set_weights(gate_up, down, router_w);
//         std::cout << "[C++] All weights set (Experts + Router)." << std::endl;
//     }
// };

// PYBIND11_MODULE(gu_moe_ops, m) {
//     py::class_<PyGuMoeWrapper>(m, "GuMoeBlock")
//         .def(py::init<int, int, int, int, int>()) 
//         .def("forward", &PyGuMoeWrapper::forward)
//         .def("set_weights", &PyGuMoeWrapper::set_weights);
// }