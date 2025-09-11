
#include "infiniop.h"
#include "utils/utils.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

// Helper function to read a binary file into a vector of floats
std::vector<float> read_bin_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read((char*)buffer.data(), size)) {
        std::cerr << "Failed to read file: " << path << std::endl;
        return {};
    }
    return buffer;
}

// Helper function to compare two vectors of floats
bool compare_vectors(const std::vector<float>& vec1, const std::vector<float>& vec2, float tolerance) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Vector sizes do not match!" << std::endl;
        return false;
    }
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::abs(vec1[i] - vec2[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << vec1[i] << " vs " << vec2[i] << std::endl;
            return false;
        }
    }
    return true;
}

int test_conv1d_prefill(InfiniopHandle handle) {
    // 2. Load test data
    auto x_data = read_bin_file("x.bin");
    auto w_data = read_bin_file("conv1d_w.bin");
    auto y_expected_data = read_bin_file("y.bin");

    if (x_data.empty() || w_data.empty() || y_expected_data.empty()) {
        return -1;
    }

    // 3. Create tensors
    InfiniopTensor x, w, y;
    int x_dims[] = {1, 64, 1, 13}; // B, C, H, W (padded)
    int w_dims[] = {64, 1, 1, 4}; // C, _, _, K
    int y_dims[] = {1, 64, 1, 10}; // B, C, H, W (output)

    infiniop_tensor_create(&x, handle, INFINIOP_DTYPE_FLOAT32, x_dims, 4, x_data.data());
    infiniop_tensor_create(&w, handle, INFINIOP_DTYPE_FLOAT32, w_dims, 4, w_data.data());
    infiniop_tensor_create(&y, handle, INFINIOP_DTYPE_FLOAT32, y_dims, 4, nullptr);

    // 4. Create and run conv1d operator
    InfiniopOperator conv1d;
    infiniop_operator_create(&conv1d, handle, INFINIOP_OP_CONV1D, nullptr);
    InfiniopTensor inputs[] = {x, w};
    InfiniopTensor outputs[] = {y};
    infiniop_operator_run(conv1d, inputs, 2, outputs, 1);

    // 5. Get output data
    std::vector<float> y_data(y_expected_data.size());
    infiniop_tensor_get_data(y, y_data.data());

    // 6. Compare results
    bool passed = compare_vectors(y_data, y_expected_data, 1e-5);
    std::cout << "Conv1D Prefill Test " << (passed ? "PASSED" : "FAILED") << std::endl;

    // 7. Clean up
    infiniop_tensor_destroy(x);
    infiniop_tensor_destroy(w);
    infiniop_tensor_destroy(y);
    infiniop_operator_destroy(conv1d);

    return passed ? 0 : -1;
}
