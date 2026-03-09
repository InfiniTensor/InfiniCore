#pragma once

#include <vector>
#include <mccl.h>

#ifdef __cplusplus
extern "C" {
#endif

// namespace custom_allreduce {

using fptr_t = int64_t;

fptr_t init_custom_ar(
    const std::vector<fptr_t>& fake_ipc_ptrs,
    void* rank_data,
    size_t rank_data_sz,
    int64_t rank,
    bool full_nvlink);

void all_reduce(
    fptr_t _fa, 
    void* inp, 
    void* out, 
    size_t rank_data_sz, 
    mcclDataType_t datatype, 
    fptr_t _reg_buffer, 
    int64_t reg_buffer_sz_bytes, 
    musaStream_t stream);

int64_t meta_size();

void register_buffer(
    fptr_t _fa,
    const std::vector<fptr_t>& fake_ipc_ptrs);

// } // namespace custom_allreduce

#ifdef __cplusplus
}
#endif