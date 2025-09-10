#include <infiniop.h>
#include <infinirt.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <random>
#include <cassert>
#include <fstream>
#include <cstring>
#include <cerrno>

static void load_tensor(const char *filename, std::vector<float>& v, size_t size) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        printf("Failed to open %s. Error: %s\n", filename, strerror(errno));
        exit(-1);
    }
    ifs.seekg(0, std::ios::end);
    size_t file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    printf("Loading %s: file_size=%zu, expected_size=%zu\n", filename, file_size, size * sizeof(float));
    v.resize(size);
    ifs.read((char *)v.data(), size * sizeof(float));
}

int main() {
    // Select device
    if (infinirtSetDevice(INFINI_DEVICE_NVIDIA, 0) != INFINI_STATUS_SUCCESS) {
        printf("No NVIDIA device or failed to set device.\n");
        return 0; // don't fail CI hard
    }

    // Shapes - 匹配qwen3_next的实际参数
    const size_t B = 1, C = 64, L = 10, K = 4;
    const size_t L_padded = L + K - 1; // Padded length
    const infiniDtype_t dtype = INFINI_DTYPE_F32;

    // Create handle
    infiniopHandle_t handle = nullptr;
    auto st = infiniopCreateHandle(&handle);
    if (st != INFINI_STATUS_SUCCESS) { printf("Create handle failed\n"); return -1; }

    // Descriptors: use 4D NCHW (H=1, W=L)
    size_t x_shape[4] = {B, C, 1, L_padded};
    size_t y_shape[4] = {B, C, 1, L};
    size_t w_shape[4] = {C, 1, 1, K};

    infiniopTensorDescriptor_t x_desc=nullptr, y_desc=nullptr, w_desc=nullptr;
    st = infiniopCreateTensorDescriptor(&x_desc, 4, x_shape, nullptr, dtype);
    if (st!=INFINI_STATUS_SUCCESS){printf("x_desc failed\n");return -1;}
    st = infiniopCreateTensorDescriptor(&y_desc, 4, y_shape, nullptr, dtype);
    if (st!=INFINI_STATUS_SUCCESS){printf("y_desc failed\n");return -1;}
    st = infiniopCreateTensorDescriptor(&w_desc, 4, w_shape, nullptr, dtype);
    if (st!=INFINI_STATUS_SUCCESS){printf("w_desc failed\n");return -1;}

    // Create op descriptor
    infiniopConv1dDescriptor_t desc = nullptr;
    st = infiniopCreateConv1dDescriptor(handle, &desc, y_desc, x_desc, w_desc, K);
    if (st != INFINI_STATUS_SUCCESS) { printf("create conv1d desc failed\n"); return -1; }

    // Workspace
    size_t ws_size=0; st = infiniopGetConv1dWorkspaceSize(desc, &ws_size);
    if (st != INFINI_STATUS_SUCCESS) { printf("get ws failed\n"); return -1; }
    void* workspace=nullptr; if (ws_size) infinirtMalloc(&workspace, ws_size);

    // Allocate device buffers
    size_t x_elems = B*C*L_padded, y_elems=B*C*L, w_elems=C*K, state_elems=B*C*(K-1);
    void *x_dev=nullptr, *y_dev=nullptr, *w_dev=nullptr, *state_dev=nullptr;
    infinirtMalloc(&x_dev, x_elems*sizeof(float));
    infinirtMalloc(&y_dev, y_elems*sizeof(float));
    infinirtMalloc(&w_dev, w_elems*sizeof(float));
    infinirtMalloc(&state_dev, state_elems*sizeof(float));

    // Host data
    std::vector<float> x_host, w_host, y_host_pytorch, y_host_infini, state_host;
    load_tensor("/home/zhujianian/workspace/zjn/InfiniCore/test/x.bin", x_host, x_elems);
    load_tensor("/home/zhujianian/workspace/zjn/InfiniCore/test/conv1d_w.bin", w_host, w_elems);
    load_tensor("/home/zhujianian/workspace/zjn/InfiniCore/test/y.bin", y_host_pytorch, y_elems);
    y_host_infini.resize(y_host_pytorch.size(), 0.0f);
    state_host.resize(B * C * (K - 1));

    // Upload
    infinirtMemcpy(x_dev, x_host.data(), x_elems*sizeof(float), INFINIRT_MEMCPY_H2D);
    infinirtMemcpy(w_dev, w_host.data(), w_elems*sizeof(float), INFINIRT_MEMCPY_H2D);

    // Run prefill
    st = infiniopConv1dFn(desc, workspace, ws_size, y_dev, x_dev, w_dev, nullptr);
    if (st != INFINI_STATUS_SUCCESS) { printf("prefill failed: %d\n", (int)st); return -1; }

    // Download a few outputs
    infinirtMemcpy(y_host_infini.data(), y_dev, y_elems*sizeof(float), INFINIRT_MEMCPY_D2H);

    // Compare results
    printf("InfiniCore x:"); for(int i=0; i<10; ++i) printf(" %.6f", x_host[i]); printf("\n");
    printf("InfiniCore w (size=%zu):", w_host.size()); for(int i=0; i<10 && i<(int)w_host.size(); ++i) printf(" %.6f", w_host[i]); printf("\n");
    printf("InfiniCore y:"); for(int i=0; i<10; ++i) printf(" %.6f", y_host_infini[i]); printf("\n");

    for (size_t i = 0; i < y_host_pytorch.size(); ++i) {
        if (std::abs(y_host_pytorch[i] - y_host_infini[i]) > 1e-5) {
            printf("Mismatch at index %zu: pytorch=%.6f, infini=%.6f\n", i, y_host_pytorch[i], y_host_infini[i]);
            return -1;
        }
    }
    printf("Prefill test passed!\n");

    // --- Test update --- //
    std::vector<float> x_now_host, conv_state_initial_host, y_update_pytorch_host, y_update_infini_host;
    load_tensor("/home/zhujianian/workspace/zjn/InfiniCore/test/x_now.bin", x_now_host, B * C);
    load_tensor("/home/zhujianian/workspace/zjn/InfiniCore/test/conv_state_initial.bin", conv_state_initial_host, B * C * (K - 1));
    load_tensor("/home/zhujianian/workspace/zjn/InfiniCore/test/y_update_pytorch.bin", y_update_pytorch_host, B * C);
    y_update_infini_host.resize(y_update_pytorch_host.size());

    void* x_now_dev=nullptr; infinirtMalloc(&x_now_dev, x_now_host.size()*sizeof(float));
    void* conv_state_dev=nullptr; infinirtMalloc(&conv_state_dev, conv_state_initial_host.size()*sizeof(float));
    void* y_now_dev=nullptr; infinirtMalloc(&y_now_dev, y_update_infini_host.size()*sizeof(float));

    infinirtMemcpy(x_now_dev, x_now_host.data(), x_now_host.size()*sizeof(float), INFINIRT_MEMCPY_H2D);
    infinirtMemcpy(conv_state_dev, conv_state_initial_host.data(), conv_state_initial_host.size()*sizeof(float), INFINIRT_MEMCPY_H2D);

    // Prepare params
    infiniopConv1dUpdateParams_t params{};
    params.desc = desc; params.y = y_now_dev; params.x_now = x_now_dev; params.w = w_dev; params.conv_state = conv_state_dev;
    params.B = B; params.C = C; params.K = K; params.dtype = dtype; params.stream = nullptr;

    st = infiniopConv1dUpdate(&params);
    if (st != INFINI_STATUS_SUCCESS) { printf("update failed: %d\n", (int)st); return -1; }

    infinirtMemcpy(y_update_infini_host.data(), y_now_dev, y_update_infini_host.size()*sizeof(float), INFINIRT_MEMCPY_D2H);

    // Compare update results
    for (size_t i = 0; i < y_update_pytorch_host.size(); ++i) {
        if (std::abs(y_update_pytorch_host[i] - y_update_infini_host[i]) > 1e-5) {
            printf("Update mismatch at index %zu: pytorch=%.6f, infini=%.6f\n", i, y_update_pytorch_host[i], y_update_infini_host[i]);
            return -1;
        }
    }
    printf("Update test passed!\n");

    // Cleanup
    if (workspace) infinirtFree(workspace);
    infinirtFree(x_dev); infinirtFree(y_dev); infinirtFree(w_dev); infinirtFree(state_dev);
    infinirtFree(x_now_dev); infinirtFree(y_now_dev); infinirtFree(conv_state_dev);

    infiniopDestroyConv1dDescriptor(desc);
    infiniopDestroyTensorDescriptor(x_desc);
    infiniopDestroyTensorDescriptor(y_desc);
    infiniopDestroyTensorDescriptor(w_desc);
    infiniopDestroyHandle(handle);

    printf("conv1d smoke passed.\n");
    return 0;
}
