#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "topk_nvidia.cuh"
#include "../cuda/kernel.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cub/cub.cuh>

namespace op::topk::nvidia {
    struct Descriptor::Opaque {
        std::shared_ptr<device::nvidia::Handle::Internal> internal;
    };
    
    Descriptor::~Descriptor() {
        delete _opaque;
    }
    
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
        auto result = TopKInfo::create(values_output_desc, indices_output_desc, input_desc, k,dim, largest, sorted);
        CHECK_RESULT(result);
        auto info = result.take();
        size_t workspace_size = 0;

        workspace_size += (input_desc->ndim() + values_output_desc->ndim()) * (sizeof(size_t) + sizeof(ptrdiff_t));
        *desc_ptr = new Descriptor(
            new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
            info, workspace_size, handle->device, handle->device_id);
        return INFINI_STATUS_SUCCESS;
    }
    
    namespace {
    
    template<size_t BLOCK_SIZE, int32_t SORT_ITEMS_PER_THREAD, typename Tdata>
    infiniStatus_t launchKernel(
        const TopKInfo &info,
        Tdata *values_output, int32_t *indices_output, const Tdata *input,
        size_t k, size_t dim, bool largest, bool sorted,
        cudaStream_t stream, void *workspace, size_t workspace_size) {
        // const int rows = (int)info.n_iteration;
        // const int n    = (int)info.dim_elements;
        // const int kk   = (int)k;
        if (dim >= info.ndim) return INFINI_STATUS_BAD_PARAM;
        if (k == 0) return INFINI_STATUS_SUCCESS;
        if (k > info.dim_elements) return INFINI_STATUS_BAD_PARAM;
        size_t input_ndim = info.ndim;
        size_t output_ndim = input_ndim;
        size_t n_iteration = info.n_iteration;
        size_t dim_elements = info.dim_elements;
        unsigned char *workspace_ptr = reinterpret_cast<unsigned char *>(workspace);
        size_t workspace_offset = 0;
        size_t *input_shape_cuda = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
        size_t *output_shape_cuda = input_shape_cuda + input_ndim;
        workspace_offset += (input_ndim + output_ndim) * sizeof(size_t);
    
        ptrdiff_t *input_strides_cuda = reinterpret_cast<ptrdiff_t *>(workspace_ptr + workspace_offset);
        ptrdiff_t *output_strides_cuda = input_strides_cuda + input_ndim;
        workspace_offset += (input_ndim + output_ndim) * sizeof(ptrdiff_t);
    
        CHECK_CUDA(cudaMemcpyAsync(input_shape_cuda,      info.input_shape.data(),     input_ndim * sizeof(size_t),      cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(output_shape_cuda,     info.output_shape.data(),    output_ndim * sizeof(size_t),     cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(input_strides_cuda,    info.input_strides.data(),   input_ndim * sizeof(ptrdiff_t),   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(output_strides_cuda,   info.output_strides.data(),  output_ndim * sizeof(ptrdiff_t),  cudaMemcpyHostToDevice, stream));
    
        const int32_t total = n_iteration * dim_elements;

        uint32_t *cur_vals = nullptr, *ones_vals = nullptr, *zeros_vals = nullptr;
        int32_t *cur_idx = nullptr, *ones_idx = nullptr, *zeros_idx = nullptr;

        uint32_t *sel_vals = nullptr, *sel_sorted_vals = nullptr;
        int32_t *sel_idx = nullptr, *sel_sorted_idx = nullptr;

        int32_t *cur_n = nullptr, *rem_k = nullptr, *out_pos = nullptr;
        int32_t *ones_count = nullptr, *zeros_count = nullptr;
        

        CHECK_CUDA(cudaMalloc(&cur_vals,   total * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&ones_vals,  total * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&zeros_vals, total * sizeof(uint32_t)));

        CHECK_CUDA(cudaMalloc(&cur_idx,   total * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&ones_idx,  total * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&zeros_idx, total * sizeof(int32_t)));

        CHECK_CUDA(cudaMalloc(&sel_vals, n_iteration * k * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&sel_idx,  n_iteration * k * sizeof(int32_t)));

        if (sorted) {
            CHECK_CUDA(cudaMalloc(&sel_sorted_vals, n_iteration * k * sizeof(uint32_t)));
            CHECK_CUDA(cudaMalloc(&sel_sorted_idx,  n_iteration * k * sizeof(int32_t)));
        }

        CHECK_CUDA(cudaMalloc(&cur_n,      n_iteration * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&rem_k,      n_iteration * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&out_pos,    n_iteration * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&ones_count, n_iteration * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&zeros_count,n_iteration * sizeof(int32_t)));
        // init
        {
            size_t threads = 256;
            size_t blocks = (n_iteration + threads - 1) / threads;
            op::topk::cuda::init_row_state<<<blocks, threads, 0, stream>>>(cur_n, rem_k, out_pos, n_iteration, dim_elements, k);
        }
        // gather input -> cur
        {
            dim3 block(BLOCK_SIZE);
            dim3 grid((dim_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, n_iteration);
            op::topk::cuda::gather_rowwise<Tdata><<<grid, block, 0, stream>>>(
                input, cur_vals, cur_idx,
                n_iteration, dim_elements,
                input_ndim, dim,
                input_shape_cuda, input_strides_cuda);
        }
        // radix select/filter
        for (int bit = 31; bit >= 0; --bit) {
            {
                size_t threads = 256;
                size_t blocks = (n_iteration + threads - 1) / threads;
                op::topk::cuda::zero_row_counters<<<blocks, threads, 0, stream>>>(ones_count, zeros_count, n_iteration);
            }
    
            {
                dim3 block(BLOCK_SIZE);
                dim3 grid((dim_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, n_iteration);
                op::topk::cuda::partition_rowwise<BLOCK_SIZE><<<grid, block, 0, stream>>>(
                    cur_vals, cur_idx,
                    ones_vals, ones_idx,
                    zeros_vals, zeros_idx,
                    cur_n, n_iteration, dim_elements,
                    bit, largest,
                    ones_count, zeros_count);
            }
    
            {
                op::topk::cuda::decide_and_compact<BLOCK_SIZE><<<n_iteration, BLOCK_SIZE, 0, stream>>>(
                    cur_vals, cur_idx,
                    ones_vals, ones_idx,
                    zeros_vals, zeros_idx,
                    ones_count, zeros_count,
                    cur_n, rem_k, out_pos,
                    sel_vals, sel_idx,
                    n_iteration, dim_elements, k);
            }
        }

        // append remaining

        op::topk::cuda::take_remaining<BLOCK_SIZE><<<n_iteration, BLOCK_SIZE, 0, stream>>>(
            cur_vals, cur_idx,
            cur_n, rem_k, out_pos,
            sel_vals, sel_idx,
            n_iteration, dim_elements, k);
    
        // optional sort (CUB block radix sort)
        // const uint32_t* final_vals = sel_vals;
        const int32_t* final_idx = sel_idx;
    
        if (sorted) {
            std::vector<int> h_offsets(n_iteration + 1);
            for(size_t i = 0; i <= n_iteration; i++){
                h_offsets[i] = i * k;
            }
            int *d_offsets;
            CHECK_CUDA(cudaMalloc(&d_offsets, (n_iteration + 1) * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), (n_iteration + 1) * sizeof(int), cudaMemcpyHostToDevice));
            
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
           
            // 或者直接给待排序的vals取反

            if (!largest) {
            

            cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx, 
                n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);

            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            
            cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx, 
                n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
            } else {
                cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx, 
                    n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
    
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                
                cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx, 
                    n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
            }
            CHECK_CUDA(cudaFree(d_offsets));
            CHECK_CUDA(cudaFree(d_temp_storage));
            // final_vals = sel_sorted_vals;
            final_idx  = sel_sorted_idx;
        }

        if (sorted &&k > 0 && n_iteration > 0) {
            // 分配主机内存来复制所有结果
            size_t total_elements = n_iteration * k;
            uint32_t *host_vals = new uint32_t[total_elements];
            int32_t *host_idx = new int32_t[total_elements];
            uint32_t *sorted_host_vals = new uint32_t[total_elements];
            int32_t *sorted_host_idx = new int32_t[total_elements];
            // 从设备复制所有输出结果
            CHECK_CUDA(cudaMemcpyAsync(sorted_host_vals, sel_sorted_vals, total_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaMemcpyAsync(sorted_host_idx, sel_sorted_idx, total_elements * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaMemcpyAsync(host_vals, sel_vals, total_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaMemcpyAsync(host_idx, sel_idx, total_elements * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            
            // 同步等待复制完成
            CHECK_CUDA(cudaStreamSynchronize(stream));
            std::cout << "k: " << k << "  dim: "<< dim << "  input_ndim: "<< input_ndim << "  n_iteration: "<< n_iteration << std::endl;
            // 打印每个 iteration 的结果
            for(size_t i = 0; i < n_iteration; i++){
                std::cout << "Iteration " << i << " (GPU):" << std::endl;
                std::cout << "  Values: ";
                for(size_t j = 0; j < k; j++){
                    size_t idx = i * k + j;
                        std::cout << host_vals[idx];
                    if (j < k - 1) std::cout << ", ";
                }
                std::cout << std::endl;
                std::cout << "  Indices: ";
                for(size_t j = 0; j < k; j++){
                    size_t idx = i * k + j;
                    std::cout << host_idx[idx];
                    if (j < k - 1) std::cout << ", ";
                }
                std::cout << std::endl << std::endl;



                std::cout << "  Sorted Values: ";
                for(size_t j = 0; j < k; j++){
                    size_t idx = i * k + j;
                    // 对于浮点数，需要转换为 float 打印，对于其他类型可以直接打印
                        std::cout << sorted_host_vals[idx];
                    if (j < k - 1) std::cout << ", ";
                }
                std::cout << std::endl;
        
                std::cout << "  Sorted Indices: ";
                for(size_t j = 0; j < k; j++){
                    size_t idx = i * k + j;
                    std::cout << sorted_host_idx[idx];
                    if (j < k - 1) std::cout << ", ";
                }
                std::cout << std::endl << std::endl;
            }
            
            delete[] host_vals;
            delete[] host_idx;
            delete[] sorted_host_vals;
            delete[] sorted_host_idx;
        }
        
    
        // scatter to output (strided write)
        {
            dim3 block(BLOCK_SIZE);
            dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, n_iteration);
            op::topk::cuda::scatter_to_output<Tdata><<<grid, block, 0, stream>>>(
                input, final_idx,
                values_output, indices_output,
                n_iteration, k,
                input_ndim, dim,
                input_shape_cuda, input_strides_cuda,
                output_shape_cuda, output_strides_cuda);
        }
    
        CHECK_CUDA(cudaGetLastError());
    

        // 添加调试打印


        // free temps
        CHECK_CUDA(cudaFree(cur_vals));
        CHECK_CUDA(cudaFree(ones_vals));
        CHECK_CUDA(cudaFree(zeros_vals));
        CHECK_CUDA(cudaFree(cur_idx));
        CHECK_CUDA(cudaFree(ones_idx));
        CHECK_CUDA(cudaFree(zeros_idx));
        CHECK_CUDA(cudaFree(sel_vals));
        CHECK_CUDA(cudaFree(sel_idx));
        if (sorted) {
            CHECK_CUDA(cudaFree(sel_sorted_vals));
            CHECK_CUDA(cudaFree(sel_sorted_idx));
        }
        CHECK_CUDA(cudaFree(cur_n));
        CHECK_CUDA(cudaFree(rem_k));
        CHECK_CUDA(cudaFree(out_pos));
        CHECK_CUDA(cudaFree(ones_count));
        CHECK_CUDA(cudaFree(zeros_count));
    
        return INFINI_STATUS_SUCCESS;
    }
        // CHECK_CUDA(cudaDeviceSynchronize());
    
} // namespace
    
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
        void *stream_) const {

            cudaStream_t stream = (cudaStream_t)stream_;
            constexpr int ITEMS = 4;
            #define CALCULATE_TOPK(BLOCK_SIZE, Tdata)                                          \
            launchKernel<BLOCK_SIZE, ITEMS, Tdata>(                                            \
                _info,                                                                         \
                (Tdata *)values_output,  (int32_t *)indices_output,  (const Tdata *)input,      \
                k, dim, largest, sorted,                                                       \
                stream, workspace, workspace_size                                              \
            )

            #define CALCULATE_TOPK_WITH_BLOCK_SIZE(BLOCK_SIZE)           \
            {                                                            \
                if (_info.dtype == INFINI_DTYPE_BF16)                    \
                    return CALCULATE_TOPK(BLOCK_SIZE, __nv_bfloat16);    \
                else if(_info.dtype == INFINI_DTYPE_F16)                 \
                    return CALCULATE_TOPK(BLOCK_SIZE, half);             \
                else if(_info.dtype == INFINI_DTYPE_F32)                 \
                    return CALCULATE_TOPK(BLOCK_SIZE, float);            \
                else                                                     \
                    return INFINI_STATUS_BAD_TENSOR_DTYPE;               \
            }

            if (_opaque->internal->maxThreadsPerBlock() >= 256) {
                CALCULATE_TOPK_WITH_BLOCK_SIZE(256)
            } else {
                return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
            }
            return INFINI_STATUS_SUCCESS;
        }
    
}