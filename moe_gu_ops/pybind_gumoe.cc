#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 必须包含，用于自动转换 map 和 string

#include "gu_moe.h" // MoE 头文件
#include "infinicore/tensor.hpp"
#include "infinicore/device.hpp"

namespace py = pybind11;

// 1. 转换器：Torch Tensor -> Infini Tensor (Zero-Copy)
// 这个函数只创建一个"视图"，不拷贝数据。
// 安全性：因为 Module::load_state_dict 内部会执行 copy_from，所以数据会被安全地拷贝到模型参数里。
infinicore::Tensor torch_to_infini_view(const torch::Tensor& t) {
    // 1. 获取形状
    infinicore::Shape shape;
    for (auto s : t.sizes()) shape.push_back(s);

    // 2. 获取数据类型 (目前代码只支持 F32)
    infinicore::DataType dtype = infinicore::DataType::F32;
    if (t.dtype() == torch::kFloat32) dtype = infinicore::DataType::F32;
    else if (t.dtype() == torch::kFloat16) dtype = infinicore::DataType::F16;
    else throw std::runtime_error("Unsupported dtype");

    // 3. 获取设备
    infinicore::Device::Type dev_type = infinicore::Device::Type::CPU;
    int dev_id = 0;
    if (t.is_cuda()) {
        dev_type = infinicore::Device::Type::NVIDIA;
        dev_id = t.device().index();
    }
    
    // 4. 创建 Tensor 视图 (from_blob)
    return infinicore::Tensor::from_blob(
        t.data_ptr(), 
        shape, 
        dtype, 
        infinicore::Device(dev_type, dev_id)
    );
}

// =====================================================================
// 2. 转换器：Infini Tensor -> Torch Tensor (用于 Forward 输出)
// =====================================================================
torch::Tensor infini_to_torch_copy(infinicore::Tensor t) {
    std::vector<int64_t> sizes;
    for (auto s : t->shape()) sizes.push_back(s);

    auto options = torch::TensorOptions().dtype(torch::kFloat32); // 假设输出 F32
    if (t->device().getType() == infinicore::Device::Type::NVIDIA) {
        options = options.device(torch::kCUDA, t->device().getIndex());
    } else {
        options = options.device(torch::kCPU);
    }
    
    // 创建并 clone，确保拥有内存
    return torch::from_blob(t->data(), sizes, options).clone();
}

// =====================================================================
// 3. 包装类 (Wrapper)
// =====================================================================
class PyGuMoeWrapper {
public:
    std::shared_ptr<infinicore::nn::GuMoeSparseMoeBlock> moe_block;

    // 构造函数：接收 Python 传来的参数
    PyGuMoeWrapper(int num_experts, int hidden_dim, int intermediate_dim, int top_k, bool norm_topk) {
        // 假设这里强制使用 NVIDIA:0，你可以根据需要添加 device 参数
        infinicore::Device device(infinicore::Device::Type::NVIDIA, 0);
        
        moe_block = std::make_shared<infinicore::nn::GuMoeSparseMoeBlock>(
            num_experts, hidden_dim, intermediate_dim, top_k, norm_topk, 
            infinicore::DataType::F32, device
        );
    }

    // Forward
    torch::Tensor forward(torch::Tensor hidden_states) {
        auto infini_input = torch_to_infini_view(hidden_states);
        auto infini_output = moe_block->forward(infini_input);
        return infini_to_torch_copy(infini_output);
    }

    // 【核心】加载权重接口
    // 接收 Python 的 Dict[str, Tensor]
    void load_state_dict(std::map<std::string, torch::Tensor> weights) {
        std::unordered_map<std::string, infinicore::Tensor> infini_dict;

        for (auto const& [name, tensor] : weights) {
            infini_dict.emplace(name, torch_to_infini_view(tensor.contiguous()));
        }
        moe_block->load_state_dict(infini_dict);
        
        std::cout << "[C++] load_state_dict finished. Loaded " << infini_dict.size() << " tensors." << std::endl;
    }
};

// =====================================================================
// 4. 定义 Python 模块
// =====================================================================
PYBIND11_MODULE(gu_moe_ops, m) {
    py::class_<PyGuMoeWrapper>(m, "GuMoeBlock")
        .def(py::init<int, int, int, int, bool>())
        .def("forward", &PyGuMoeWrapper::forward)
        .def("load_state_dict", &PyGuMoeWrapper::load_state_dict);
}

