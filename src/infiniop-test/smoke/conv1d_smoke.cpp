#include <infiniop.h>
#include <infinirt.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <random>
#include <cassert>

int main() {
    // Select device
    if (infinirtSetDevice(INFINI_DEVICE_NVIDIA, 0) != INFINI_STATUS_SUCCESS) {
        printf("No NVIDIA device or failed to set device.\n");
        return 0; // don't fail CI hard
    }

    // Shapes for 1D convolution: [B, Cin, L]
    const size_t B = 2, Cin = 8, L = 16, Cout = 12, K = 3;
    const infiniDtype_t dtype = INFINI_DTYPE_F32;

    // Create handle
    infiniopHandle_t handle = nullptr;
    auto st = infiniopCreateHandle(&handle);
    if (st != INFINI_STATUS_SUCCESS) { printf("Create handle failed\n"); return -1; }

    // Create tensor descriptors for 1D conv: [B, C, L]
    size_t x_shape[3] = {B, Cin, L};
    size_t y_shape[3] = {B, Cout, L}; // output length same as input for padding=K-1
    size_t w_shape[3] = {Cout, Cin, K};

    infiniopTensorDescriptor_t x_desc=nullptr, y_desc=nullptr, w_desc=nullptr;
    st = infiniopCreateTensorDescriptor(&x_desc, 3, x_shape, nullptr, dtype);
    if (st!=INFINI_STATUS_SUCCESS){printf("x_desc failed\n");return -1;}
    st = infiniopCreateTensorDescriptor(&y_desc, 3, y_shape, nullptr, dtype);
    if (st!=INFINI_STATUS_SUCCESS){printf("y_desc failed\n");return -1;}
    st = infiniopCreateTensorDescriptor(&w_desc, 3, w_shape, nullptr, dtype);
    if (st!=INFINI_STATUS_SUCCESS){printf("w_desc failed\n");return -1;}

    // Create conv1d descriptor
    infiniopConv1dDescriptor_t desc = nullptr;
    size_t pads[] = {K-1}; // padding for causal conv1d
    size_t strides[] = {1};
    size_t dilations[] = {1};
    st = infiniopCreateConv1dDescriptor(handle, &desc, y_desc, x_desc, w_desc, nullptr, 
                                       pads, strides, dilations, 1);
    if (st != INFINI_STATUS_SUCCESS) { printf("create conv1d desc failed\n"); return -1; }

    // Get workspace size
    size_t ws_size=0; 
    st = infiniopGetConv1dWorkspaceSize(desc, &ws_size);
    if (st != INFINI_STATUS_SUCCESS) { printf("get workspace size failed\n"); return -1; }
    void* workspace=nullptr; 
    if (ws_size) infinirtMalloc(&workspace, ws_size);

    // Allocate device buffers
    size_t x_elems = B*Cin*L, y_elems=B*Cout*L, w_elems=Cout*Cin*K;
    void *x_dev=nullptr, *y_dev=nullptr, *w_dev=nullptr;
    infinirtMalloc(&x_dev, x_elems*sizeof(float));
    infinirtMalloc(&y_dev, y_elems*sizeof(float));
    infinirtMalloc(&w_dev, w_elems*sizeof(float));

    // Generate random test data
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> x_host(x_elems), w_host(w_elems), y_host_infini(y_elems);
    for (auto& val : x_host) val = dis(gen);
    for (auto& val : w_host) val = dis(gen);

    // Upload data to device
    infinirtMemcpy(x_dev, x_host.data(), x_elems*sizeof(float), INFINIRT_MEMCPY_H2D);
    infinirtMemcpy(w_dev, w_host.data(), w_elems*sizeof(float), INFINIRT_MEMCPY_H2D);

    // Run conv1d
    st = infiniopConv1d(desc, workspace, ws_size, y_dev, x_dev, w_dev, nullptr, nullptr);
    if (st != INFINI_STATUS_SUCCESS) { printf("conv1d failed: %d\n", (int)st); return -1; }

    // Download outputs
    infinirtMemcpy(y_host_infini.data(), y_dev, y_elems*sizeof(float), INFINIRT_MEMCPY_D2H);

    // Print some results
    printf("Conv1d completed successfully!\n");
    printf("Input shapes: x[%zu,%zu,%zu], w[%zu,%zu,%zu], y[%zu,%zu,%zu]\n", 
           B, Cin, L, Cout, Cin, K, B, Cout, L);
    printf("Input x (first 5):  "); for(int i=0; i<5; ++i) printf(" %.6f", x_host[i]); printf("\n");
    printf("Weight w (first 5): "); for(int i=0; i<5; ++i) printf(" %.6f", w_host[i]); printf("\n");
    printf("Output y (first 5): "); for(int i=0; i<5; ++i) printf(" %.6f", y_host_infini[i]); printf("\n");

    // Basic sanity check - output should not be all zeros
    bool has_nonzero = false;
    for (size_t i = 0; i < y_elems; ++i) {
        if (std::abs(y_host_infini[i]) > 1e-8) {
            has_nonzero = true;
            break;
        }
    }
    if (!has_nonzero) {
        printf("ERROR: All outputs are zero!\n");
        return -1;
    }
    printf("Conv1d smoke test passed!\n");

    // Cleanup
    if (workspace) infinirtFree(workspace);
    infinirtFree(x_dev); infinirtFree(y_dev); infinirtFree(w_dev);

    infiniopDestroyConv1dDescriptor(desc);
    infiniopDestroyTensorDescriptor(x_desc);
    infiniopDestroyTensorDescriptor(y_desc);
    infiniopDestroyTensorDescriptor(w_desc);
    infiniopDestroyHandle(handle);

    printf("conv1d smoke test completed successfully.\n");
    return 0;
}