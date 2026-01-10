#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "infiniop/handle.h"
#include "infiniop/tensor_descriptor.h"
#include "infiniop/ops/hardtanh.h"

#define CHECK_INFINI(op)                                                                           \
    do {                                                                                           \
        infiniStatus_t status = (op);                                                              \
        if (status != INFINI_STATUS_SUCCESS) {                                                     \
            std::cerr << "Infiniop error at " << __FILE__ << ":" << __LINE__ << " -> " << status \
                      << std::endl;                                                               \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

#define CHECK_CUDA(op)                                                                             \
    do {                                                                                           \
        cudaError_t err = (op);                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "              \
                      << cudaGetErrorString(err) << std::endl;                                     \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main() {
    CHECK_CUDA(cudaSetDevice(0));

    infiniopHandle_t handle;
    CHECK_INFINI(infiniopCreateHandle(&handle));

    const size_t ndim = 2;
    size_t shape[ndim] = {13, 4};
    infiniopTensorDescriptor_t input_desc;
    infiniopTensorDescriptor_t output_desc;
    CHECK_INFINI(infiniopCreateTensorDescriptor(&input_desc, ndim, shape, nullptr, INFINI_DTYPE_F32));
    CHECK_INFINI(infiniopCreateTensorDescriptor(&output_desc, ndim, shape, nullptr, INFINI_DTYPE_F32));

    size_t numel = shape[0] * shape[1];
    std::vector<float> host_input(numel);
    for (size_t i = 0; i < numel; ++i) {
        host_input[i] = static_cast<float>(i) / 10.f - 2.f;
    }

    float *d_input = nullptr;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, numel * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, numel * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, host_input.data(), numel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, numel * sizeof(float)));

    infiniopHardTanhDescriptor_t desc = nullptr;
    CHECK_INFINI(infiniopCreateHardTanhDescriptor(
        handle, &desc, output_desc, input_desc, -1.0f, 1.0f));

    size_t workspace_size = 0;
    CHECK_INFINI(infiniopGetHardTanhWorkspaceSize(desc, &workspace_size));

    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    std::cout << "Workspace bytes: " << workspace_size << std::endl;

    infiniStatus_t status = infiniopHardTanh(
        desc, workspace, workspace_size, d_output, d_input, nullptr);
    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "infiniopHardTanh failed with status " << status << std::endl;
        return 1;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> host_output(numel);
    CHECK_CUDA(cudaMemcpy(host_output.data(), d_output, numel * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (size_t i = 0; i < numel; ++i) {
        float ref = std::max(-1.0f, std::min(1.0f, host_input[i]));
        max_err = std::max(max_err, std::abs(ref - host_output[i]));
    }

    std::cout << "Max abs error: " << max_err << std::endl;

    if (workspace) {
        cudaFree(workspace);
    }
    CHECK_INFINI(infiniopDestroyHardTanhDescriptor(desc));
    CHECK_INFINI(infiniopDestroyTensorDescriptor(input_desc));
    CHECK_INFINI(infiniopDestroyTensorDescriptor(output_desc));
    CHECK_INFINI(infiniopDestroyHandle(handle));
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Done" << std::endl;
    return 0;
}
