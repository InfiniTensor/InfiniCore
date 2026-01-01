#include "topk_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../../utils.h"
#include <algorithm>
#include <vector>
namespace op::topk::cpu {

Descriptor::~Descriptor() {}
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t values_output_desc,
    infiniopTensorDescriptor_t indices_output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t k,
    size_t dim, 
    bool largest,
    bool sorted) {
    // auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = TopKInfo::create(values_output_desc, indices_output_desc, input_desc, k, dim, largest, sorted);
    CHECK_RESULT(result);
    
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace{
template<typename Tdata>
infiniStatus_t calculateTopK(
    const TopKInfo &info,
    Tdata *values_output,
    int32_t *indices_output,
    const Tdata *input,
    size_t k,
    size_t dim,
    bool largest,
    bool sorted){
        if (k == 0) {
            return INFINI_STATUS_SUCCESS;
        }
        for(size_t i = 0; i < info.n_iteration; i++){
            size_t index = i;
            size_t input_start = 0;
            size_t output_start = 0;
            for(int j = info.ndim - 1; j >= 0; j--){
                if(j == (int)dim){continue;}
                input_start += (index % info.input_shape[j]) * info.input_strides[j];
                output_start += (index % info.output_shape[j]) * info.output_strides[j];
                index /= info.input_shape[j];
            }
            using elem_t = std::pair<Tdata, size_t>;
            std::vector<elem_t> vi_queue(info.dim_elements);
            for(size_t j = 0; j < info.dim_elements; j++){
                vi_queue[j].first = input[input_start + j * info.input_strides[dim]];
                vi_queue[j].second = j;
            }
            // 需增加isnan的判断
            bool use_partial_sort = static_cast<size_t>(k) * 64 <= info.dim_elements;

            if (use_partial_sort) {
                // 使用 partial_sort
                if(largest){
                    std::partial_sort(vi_queue.begin(), vi_queue.begin() + k, vi_queue.end(), 
                                    [](const elem_t &a, const elem_t &b) -> bool {
                                        return utils::cast<float>(a.first) > utils::cast<float>(b.first);
                                    });
                } else {
                    std::partial_sort(vi_queue.begin(), vi_queue.begin() + k, vi_queue.end(), 
                                    [](const elem_t &a, const elem_t &b) -> bool {
                                        return utils::cast<float>(a.first) < utils::cast<float>(b.first);
                                    });
                }
            } else {
                // 使用 nth_element + sort (当 sorted=true 时)
                if(largest){
                    std::nth_element(vi_queue.begin(), vi_queue.begin() + k - 1, vi_queue.end(), 
                                    [](const elem_t &a, const elem_t &b) -> bool {
                                        return utils::cast<float>(a.first) > utils::cast<float>(b.first);
                                    });
                    if (sorted) {
                        std::sort(vi_queue.begin(), vi_queue.begin() + k,  // 注意：PyTorch 这里是 k，不是 k-1
                                [](const elem_t &a, const elem_t &b) -> bool {
                                    return utils::cast<float>(a.first) > utils::cast<float>(b.first);
                                });
                    }
                } else {
                    std::nth_element(vi_queue.begin(), vi_queue.begin() + k - 1, vi_queue.end(), 
                                    [](const elem_t &a, const elem_t &b) -> bool {
                                        return utils::cast<float>(a.first) < utils::cast<float>(b.first);
                                    });
                    if (sorted) {
                        std::sort(vi_queue.begin(), vi_queue.begin() + k,  // 注意：PyTorch 这里是 k，不是 k-1
                                [](const elem_t &a, const elem_t &b) -> bool {
                                    return utils::cast<float>(a.first) < utils::cast<float>(b.first);
                                });
                    }
                }
            }
            for(size_t j = 0; j < k; j++){
                values_output[output_start + j * info.output_strides[dim]] = vi_queue[j].first;
                indices_output[output_start + j * info.output_strides[dim]] = (int32_t)vi_queue[j].second;
            }
            // 添加调试打印
            // std::cout << "Iteration " << i << ":" << std::endl;
            // std::cout << "  Values: ";
            // for(size_t j = 0; j < k; j++){
            //     // 对于浮点数，需要转换为 float 打印，对于其他类型可以直接打印
            //     if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
            //         std::cout << utils::cast<float>(vi_queue[j].first);
            //     } else {
            //         std::cout << vi_queue[j].first;
            //     }
            //     if (j < k - 1) std::cout << ", ";
            // }
            // std::cout << std::endl;

            // std::cout << "  Indices: ";
            // for(size_t j = 0; j < k; j++){
            //     std::cout << vi_queue[j].second;
            //     if (j < k - 1) std::cout << ", ";
            // }
            // std::cout << std::endl << std::endl;
        }
        return INFINI_STATUS_SUCCESS;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *values_output,
    void *indices_output,
    const void *input,
    size_t k,
    size_t dim,
    bool largest,
    bool sorted,
    void *stream) const {
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return calculateTopK<fp16_t>(_info, (fp16_t *)values_output, (int32_t *)indices_output, reinterpret_cast<const fp16_t *>(input), k, dim, largest, sorted);
    case INFINI_DTYPE_F32:
        return calculateTopK<float>(_info, (float *)values_output, (int32_t *)indices_output, reinterpret_cast<const float *>(input), k, dim, largest, sorted);
    case INFINI_DTYPE_BF16:
        return calculateTopK<bf16_t>(_info, (bf16_t *)values_output, (int32_t *)indices_output, reinterpret_cast<const bf16_t *>(input), k, dim, largest, sorted);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::topk::cpu
