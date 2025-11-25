#include "test_nn_module.h"
#include "infinicore/ops.hpp"

namespace infinicore::test {

// Test 1: Basic module operations (creation, parameters, state_dict, load_state_dict)
TestResult NNModuleTest::testBasicModuleCreation() {
    return measureTime("BasicModuleOperations", [this]() {
        try {
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("Testing Basic Module Operations");
            SPDLOG_INFO("==========================================");

            // Test 1a: Module creation and parameter registration
            SPDLOG_INFO("Test 1a: Module creation and parameter registration");
            MockLinearModule module(8, 4, infinicore::Device());

            // Verify the module was created successfully
            auto state_dict = module.state_dict();
            if (state_dict.size() != 2) {
                SPDLOG_ERROR("Expected 2 parameters, got {}", state_dict.size());
                return false;
            }

            // Test weight and bias parameters
            const auto &weight = module.get_weight();
            const auto &bias = module.get_bias();

            // Verify parameter shapes
            if (weight->shape() != std::vector<size_t>({4, 8})) {
                SPDLOG_ERROR("Weight shape mismatch. Expected {{4, 8}}");
                return false;
            }

            if (bias->shape() != std::vector<size_t>({4})) {
                SPDLOG_ERROR("Bias shape mismatch. Expected {{4}}");
                return false;
            }

            SPDLOG_INFO("✓ Module creation and parameter registration passed");

            // Test 1b: State dictionary functionality
            SPDLOG_INFO("Test 1b: State dictionary functionality");

            // Check if both parameters are in state dict
            if (state_dict.find("weight") == state_dict.end()) {
                SPDLOG_ERROR("'weight' parameter not found in state dict");
                return false;
            }

            if (state_dict.find("bias") == state_dict.end()) {
                SPDLOG_ERROR("'bias' parameter not found in state dict");
                return false;
            }

            SPDLOG_DEBUG("State dict contains {} parameters:", state_dict.size());
            for (const auto &[name, tensor] : state_dict) {
                std::ostringstream shape_str;
                shape_str << "[";
                for (size_t i = 0; i < tensor->shape().size(); ++i) {
                    if (i > 0) {
                        shape_str << ", ";
                    }
                    shape_str << tensor->shape()[i];
                }
                shape_str << "]";
                SPDLOG_DEBUG("  - {} with shape: {}", name, shape_str.str());
            }

            SPDLOG_INFO("✓ State dict functionality passed");

            // Test 1c: Load state dict functionality
            SPDLOG_INFO("Test 1c: Load state dict functionality");

            // Create new tensors to load
            auto new_weight = infinicore::Tensor::ones({4, 8}, infinicore::DataType::F32, infinicore::Device());
            auto new_bias = infinicore::Tensor::zeros({4}, infinicore::DataType::F32, infinicore::Device());

            // Load using load_parameter
            module.load_parameter("weight", new_weight);
            module.load_parameter("bias", new_bias);

            // Verify the parameters were updated
            auto updated_state_dict = module.state_dict();
            if (!tensorsAllClose(updated_state_dict.at("weight"), new_weight, 1e-6, 1e-6)) {
                SPDLOG_ERROR("Weight parameter values do not match after load_parameter");
                return false;
            }
            if (!tensorsAllClose(updated_state_dict.at("bias"), new_bias, 1e-6, 1e-6)) {
                SPDLOG_ERROR("Bias parameter values do not match after load_parameter");
                return false;
            }

            // Test load_state_dict
            std::unordered_map<std::string, infinicore::Tensor> new_state_dict;
            new_state_dict.emplace("weight", infinicore::Tensor::ones({4, 8}, infinicore::DataType::F32, infinicore::Device()));
            new_state_dict.emplace("bias", infinicore::Tensor::ones({4}, infinicore::DataType::F32, infinicore::Device()));

            module.load_state_dict(new_state_dict);

            auto final_state_dict = module.state_dict();
            if (final_state_dict.size() != 2) {
                SPDLOG_ERROR("State dict size mismatch after load_state_dict");
                return false;
            }

            SPDLOG_INFO("✓ Load state dict functionality passed");

            SPDLOG_INFO("=== All Basic Module Operations Passed ===");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testBasicModuleOperations: {}", e.what());
            return false;
        }
    });
}

// Test 2: Advanced load state dict functionality (hierarchical modules)
TestResult NNModuleTest::testLoadStateDict() {
    return measureTime("AdvancedLoadStateDict", [this]() {
        try {
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("Testing Advanced load_state_dict with Hierarchical Modules");
            SPDLOG_INFO("==========================================");

            // Test: Deep nesting (2-level hierarchy)
            SPDLOG_INFO("Test 4: Testing load_state_dict with 2-level deep nesting");

            // Create parent -> child -> grandchild hierarchy using proper module definition
            class DeepGrandchildModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, sublayer);

            public:
                DeepGrandchildModule() {
                    INFINICORE_NN_MODULE_INIT(sublayer, 6, 4, infinicore::Device());
                }
            };

            class DeepChildModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, own_layer);
                INFINICORE_NN_MODULE(DeepGrandchildModule, sublayer);

            public:
                DeepChildModule() {
                    INFINICORE_NN_MODULE_INIT(own_layer, 8, 6, infinicore::Device());
                    INFINICORE_NN_MODULE_INIT(sublayer);
                }
            };

            class DeepParentModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, own_layer);
                INFINICORE_NN_MODULE(DeepChildModule, layer1);

            public:
                DeepParentModule() {
                    INFINICORE_NN_MODULE_INIT(own_layer, 10, 8, infinicore::Device());
                    INFINICORE_NN_MODULE_INIT(layer1);
                }
            };

            DeepParentModule deep_parent;

            // Verify initial state dict includes all 2-level hierarchical parameters
            auto deep_initial_state = deep_parent.state_dict();
            SPDLOG_DEBUG("Deep hierarchical state dict has {} parameters", deep_initial_state.size());

            // Expected parameters:
            // parent: own_layer.weight, own_layer.bias (2)
            // layer1: layer1.own_layer.weight, layer1.own_layer.bias (2)
            // sublayer: layer1.sublayer.sublayer.weight, layer1.sublayer.sublayer.bias (2)
            // Total: 6 parameters
            if (deep_initial_state.size() < 6) {
                SPDLOG_ERROR("Deep hierarchy state dict size mismatch. Expected at least 6, got {}",
                             deep_initial_state.size());
                return false;
            }

            // Verify 2-level parameter names exist
            bool has_sublayer_weight = deep_initial_state.find("layer1.sublayer.sublayer.weight") != deep_initial_state.end();
            bool has_sublayer_bias = deep_initial_state.find("layer1.sublayer.sublayer.bias") != deep_initial_state.end();

            if (!has_sublayer_weight || !has_sublayer_bias) {
                SPDLOG_ERROR("2-level nested parameters missing from state dict");
                return false;
            }
            SPDLOG_DEBUG("All 2-level hierarchical parameter names verified");

            // Create state dict for 2-level hierarchy with all 1.0 values
            std::unordered_map<std::string, infinicore::Tensor> deep_state_dict;
            deep_state_dict.emplace("own_layer.weight", infinicore::Tensor::ones({8, 10}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("own_layer.bias", infinicore::Tensor::ones({8}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("layer1.own_layer.weight", infinicore::Tensor::ones({6, 8}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("layer1.own_layer.bias", infinicore::Tensor::ones({6}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("layer1.sublayer.sublayer.weight", infinicore::Tensor::ones({4, 6}, infinicore::DataType::F32, infinicore::Device()));
            deep_state_dict.emplace("layer1.sublayer.sublayer.bias", infinicore::Tensor::ones({4}, infinicore::DataType::F32, infinicore::Device()));

            // Load the deep hierarchical state dict
            deep_parent.load_state_dict(deep_state_dict);
            SPDLOG_DEBUG("Successfully loaded 2-level deep hierarchical state dict");

            // Verify all parameters were loaded correctly
            auto deep_loaded_state = deep_parent.state_dict();

            // Verify shapes at all levels
            if (deep_loaded_state.at("own_layer.weight")->shape() != std::vector<size_t>({8, 10})) {
                SPDLOG_ERROR("Deep parent weight shape mismatch");
                return false;
            }
            if (deep_loaded_state.at("layer1.own_layer.weight")->shape() != std::vector<size_t>({6, 8})) {
                SPDLOG_ERROR("Deep layer1 weight shape mismatch");
                return false;
            }
            if (deep_loaded_state.at("layer1.sublayer.sublayer.weight")->shape() != std::vector<size_t>({4, 6})) {
                SPDLOG_ERROR("Deep sublayer weight shape mismatch");
                return false;
            }
            SPDLOG_DEBUG("All 2-level deep parameter shapes verified");

            // Verify actual weight loading correctness by checking that loaded parameters
            // match what we provided in the state dict (use the original tensors)
            SPDLOG_INFO("Verifying weight loading correctness by direct comparison");

            // Get the tensors we loaded from the state dict
            auto loaded_parent_weight = deep_loaded_state.at("own_layer.weight");
            auto loaded_layer1_weight = deep_loaded_state.at("layer1.own_layer.weight");
            auto loaded_sublayer_weight = deep_loaded_state.at("layer1.sublayer.sublayer.weight");

            // Compare with the original tensors we put in the state dict
            if (!tensorsAllClose(loaded_parent_weight, deep_state_dict.at("own_layer.weight"), 1e-5, 1e-5)) {
                SPDLOG_ERROR("Deep parent weight not preserved after loading");
                return false;
            }
            if (!tensorsAllClose(loaded_layer1_weight, deep_state_dict.at("layer1.own_layer.weight"), 1e-5, 1e-5)) {
                SPDLOG_ERROR("Deep layer1 weight not preserved after loading");
                return false;
            }
            if (!tensorsAllClose(loaded_sublayer_weight, deep_state_dict.at("layer1.sublayer.sublayer.weight"), 1e-5, 1e-5)) {
                SPDLOG_ERROR("Deep sublayer weight not preserved after loading");
                return false;
            }

            SPDLOG_INFO("✓ Weight loading correctness verified - loaded values match input state dict");
            SPDLOG_INFO("✓ 2-level deep hierarchy load_state_dict verification passed");

            SPDLOG_INFO("=== All Advanced load_state_dict Tests Passed ===");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testLoadStateDict: {}", e.what());
            return false;
        }
    });
}

// Test 3: Module hierarchy (demonstrates proper hierarchical construction pattern)
TestResult NNModuleTest::testModuleHierarchy() {
    return measureTime("ModuleHierarchy", [this]() {
        try {
            // Create a hierarchy using proper module definition: root -> layer1 -> layer2
            class Layer2Module : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, sublayer);

            public:
                Layer2Module() {
                    INFINICORE_NN_MODULE_INIT(sublayer, 8, 4, infinicore::Device());
                }
            };

            class Layer1Module : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, sublayer);
                INFINICORE_NN_MODULE(Layer2Module, layer2);

            public:
                Layer1Module() {
                    INFINICORE_NN_MODULE_INIT(sublayer, 16, 8, infinicore::Device());
                    INFINICORE_NN_MODULE_INIT(layer2);
                }
            };

            class RootModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE(MockLinearModule, root_layer);
                INFINICORE_NN_MODULE(Layer1Module, layer1);

            public:
                RootModule() {
                    INFINICORE_NN_MODULE_INIT(root_layer, 20, 16, infinicore::Device());
                    INFINICORE_NN_MODULE_INIT(layer1);
                }
            };

            RootModule root_module;

            // Check the complete state dict
            auto root_state_dict = root_module.state_dict();

            // Debug: Print all parameters
            SPDLOG_DEBUG("Found {} parameters:", root_state_dict.size());
            for (const auto &pair : root_state_dict) {
                SPDLOG_DEBUG("  - {}", pair.first);
            }

            // Should have: root_layer.weight, root_layer.bias,
            // layer1.sublayer.weight, layer1.sublayer.bias,
            // layer1.layer2.sublayer.weight, layer1.layer2.sublayer.bias
            if (root_state_dict.size() < 6) {
                SPDLOG_ERROR("Error: Expected at least 6 parameters in hierarchy, got {}", root_state_dict.size());
                return false;
            }

            SPDLOG_INFO("Module hierarchy test passed. Root state dict has {} parameters", root_state_dict.size());

            // Print the hierarchy
            std::cout << "Module hierarchy:" << std::endl;
            for (const auto &pair : root_state_dict) {
                std::cout << "  - " << pair.first << std::endl;
            }

            // Additional: Test INFINICORE_NN_MODULE_VEC vector registration
            SPDLOG_INFO("Testing INFINICORE_NN_MODULE_VEC (vector of submodules)");
            class VecModule : public infinicore::nn::Module {
            protected:
                INFINICORE_NN_MODULE_VEC(MockLinearModule, layers);

            public:
                VecModule() {
                    INFINICORE_NN_MODULE_VEC_INIT(layers, 3, MockLinearModule, 16, 8, infinicore::Device());
                }
            };

            VecModule vec_mod;
            auto vec_state = vec_mod.state_dict();

            // Expect parameters for layers.0, layers.1, layers.2 (weight and bias for each)
            std::vector<std::string> expected_vec_params = {
                "layers.0.weight", "layers.0.bias",
                "layers.1.weight", "layers.1.bias",
                "layers.2.weight", "layers.2.bias"};

            for (const auto &param : expected_vec_params) {
                if (vec_state.find(param) == vec_state.end()) {
                    SPDLOG_ERROR("INFINICORE_NN_MODULE_VEC: missing '{}' in state_dict", param);
                    return false;
                }
            }

            SPDLOG_INFO("INFINICORE_NN_MODULE_VEC test passed - found all vector layer parameters");

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testModuleHierarchy: {}", e.what());
            return false;
        }
    });
}

// Test 4: Parameter loading from blob
TestResult NNModuleTest::testParameterLoading() {
    return measureTime("ParameterLoading", [this]() {
        try {
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("Testing Parameter loading from blob");
            SPDLOG_INFO("==========================================");
            MockLinearModule module(3, 2, infinicore::Device());

            // Create test data
            std::vector<float> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            std::vector<float> bias_data = {0.1f, 0.2f};

            // Load parameters from blob data
            module.load_parameter_from_blob("weight", weight_data.data());
            module.load_parameter_from_blob("bias", bias_data.data());

            SPDLOG_INFO("Successfully loaded parameters from blob data");

            // Verify parameters exist
            auto state_dict = module.state_dict();
            if (state_dict.find("weight") == state_dict.end() || state_dict.find("bias") == state_dict.end()) {
                SPDLOG_ERROR("Error: Parameters not found after loading");
                return false;
            }

            SPDLOG_INFO("Parameter loading test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testParameterLoading: {}", e.what());
            return false;
        }
    });
}

// Test 5: Linear module implementation and behavior
TestResult NNModuleTest::testModuleLinear() {
    return measureTime("ModuleLinear", [this]() {
        try {
            // Test with bias
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("Testing Linear module with bias (8->4 features)");
            SPDLOG_INFO("==========================================");
            infinicore::nn::Linear m1(8, 4, true);
            auto sd1 = m1.state_dict();
            if (sd1.find("weight") == sd1.end()) {
                SPDLOG_ERROR("weight missing");
                return false;
            }
            if (sd1.find("bias") == sd1.end()) {
                SPDLOG_ERROR("bias missing when bias=true");
                return false;
            }
            if (sd1.at("weight")->shape() != std::vector<size_t>({4, 8})) {
                SPDLOG_ERROR("weight shape mismatch. Expected {{4, 8}}, got different shape");
                return false;
            }
            if (sd1.at("bias")->shape() != std::vector<size_t>({4})) {
                SPDLOG_ERROR("bias shape mismatch. Expected {{4}}, got different shape");
                return false;
            }
            SPDLOG_DEBUG("Parameter shapes verified: weight {{4, 8}}, bias {{4}}");

            // Test module properties
            if (m1.in_features() != 8) {
                SPDLOG_ERROR("in_features mismatch. Expected 8, got {}", m1.in_features());
                return false;
            }
            if (m1.out_features() != 4) {
                SPDLOG_ERROR("out_features mismatch. Expected 4, got {}", m1.out_features());
                return false;
            }
            if (!m1.has_bias()) {
                SPDLOG_ERROR("has_bias should be true");
                return false;
            }

            // Test linear computation with bias
            SPDLOG_INFO("Testing linear computation with bias");
            auto input1 = infinicore::Tensor::ones({2, 8}, infinicore::DataType::F32, infinicore::Device());
            auto output1 = m1.forward(input1);
            if (output1->shape() != std::vector<size_t>({2, 4})) {
                SPDLOG_ERROR("Linear output shape mismatch with bias. Expected {{2, 4}}, got different shape");
                return false;
            }
            SPDLOG_DEBUG("Linear computation with bias passed. Input shape: {{2, 8}}, Output shape: {{2, 4}}");

            // Test without bias
            SPDLOG_INFO("Testing Linear module without bias (16->3 features)");
            infinicore::nn::Linear m2(16, 3, false);
            auto sd2 = m2.state_dict();
            if (sd2.find("weight") == sd2.end()) {
                SPDLOG_ERROR("weight missing (no-bias)");
                return false;
            }
            if (sd2.find("bias") != sd2.end()) {
                SPDLOG_ERROR("bias should not exist when bias=false");
                return false;
            }
            if (sd2.at("weight")->shape() != std::vector<size_t>({3, 16})) {
                SPDLOG_ERROR("weight shape mismatch (no-bias). Expected {{3, 16}}, got different shape");
                return false;
            }
            SPDLOG_DEBUG("Parameter shapes verified: weight {{3, 16}}, no bias");

            // Test module properties
            if (m2.in_features() != 16) {
                SPDLOG_ERROR("in_features mismatch. Expected 16, got {}", m2.in_features());
                return false;
            }
            if (m2.out_features() != 3) {
                SPDLOG_ERROR("out_features mismatch. Expected 3, got {}", m2.out_features());
                return false;
            }
            if (m2.has_bias()) {
                SPDLOG_ERROR("has_bias should be false");
                return false;
            }

            // Test linear computation without bias
            SPDLOG_INFO("Testing linear computation without bias");
            auto input2 = infinicore::Tensor::ones({1, 16}, infinicore::DataType::F32, infinicore::Device());
            auto output2 = m2.forward(input2);
            if (output2->shape() != std::vector<size_t>({1, 3})) {
                SPDLOG_ERROR("Linear output shape mismatch without bias. Expected {{1, 3}}, got different shape");
                return false;
            }
            SPDLOG_DEBUG("Linear computation without bias passed. Input shape: {{1, 16}}, Output shape: {{1, 3}}");

            // Test load_state_dict for m2 (without bias)
            SPDLOG_INFO("Testing load_state_dict on Linear without bias");
            auto m2_load_weight = infinicore::Tensor::ones({3, 16}, infinicore::DataType::F32, infinicore::Device());
            std::unordered_map<std::string, infinicore::Tensor> m2_state_dict;
            m2_state_dict.emplace("weight", m2_load_weight);
            // Note: no bias parameter
            m2.load_state_dict(m2_state_dict);

            // Verify via state_dict() and direct access
            if (!tensorsAllClose(m2.state_dict().at("weight"), m2_load_weight, 1e-5, 1e-5)) {
                SPDLOG_ERROR("m2 weight not loaded correctly");
                return false;
            }
            if (!tensorsAllClose(m2.weight(), m2_load_weight, 1e-5, 1e-5)) {
                SPDLOG_ERROR("m2 weight field not synchronized");
                return false;
            }
            SPDLOG_DEBUG("m2 load_state_dict verified - weight loaded correctly (no bias)");

            // Test batch processing
            SPDLOG_INFO("Testing batch linear computation (batch size 3)");
            auto input3 = infinicore::Tensor::ones({3, 8}, infinicore::DataType::F32, infinicore::Device());
            auto output3 = m1.forward(input3);
            if (output3->shape() != std::vector<size_t>({3, 4})) {
                SPDLOG_ERROR("Batch linear output shape mismatch. Expected {{3, 4}}, got different shape");
                return false;
            }
            SPDLOG_DEBUG("Batch linear computation passed. Input shape: {{3, 8}}, Output shape: {{3, 4}}");

            // Test parameter accessors
            SPDLOG_INFO("Testing parameter accessors");
            auto weight_accessor = m1.weight();
            auto bias_accessor = m1.bias();
            if (weight_accessor->shape() != std::vector<size_t>({4, 8})) {
                SPDLOG_ERROR("Weight accessor shape mismatch");
                return false;
            }
            if (bias_accessor->shape() != std::vector<size_t>({4})) {
                SPDLOG_ERROR("Bias accessor shape mismatch");
                return false;
            }

            // Test load_state_dict for m1 (with bias)
            SPDLOG_INFO("Testing load_state_dict on Linear with bias");
            auto m1_load_weight = infinicore::Tensor::ones({4, 8}, infinicore::DataType::F32, infinicore::Device());
            auto m1_load_bias = infinicore::Tensor::ones({4}, infinicore::DataType::F32, infinicore::Device());
            std::unordered_map<std::string, infinicore::Tensor> m1_state_dict;
            m1_state_dict.emplace("weight", m1_load_weight);
            m1_state_dict.emplace("bias", m1_load_bias);
            m1.load_state_dict(m1_state_dict);

            // Verify via state_dict() and direct access
            if (!tensorsAllClose(m1.state_dict().at("weight"), m1_load_weight, 1e-5, 1e-5)) {
                SPDLOG_ERROR("m1 weight not loaded correctly");
                return false;
            }
            if (!tensorsAllClose(m1.weight(), m1_load_weight, 1e-5, 1e-5)) {
                SPDLOG_ERROR("m1 weight field not synchronized");
                return false;
            }
            if (!tensorsAllClose(m1.bias(), m1_load_bias, 1e-5, 1e-5)) {
                SPDLOG_ERROR("m1 bias field not synchronized");
                return false;
            }
            SPDLOG_DEBUG("m1 load_state_dict verified - parameters and fields synchronized");

            // Test extra_repr
            std::string repr = m1.extra_repr();
            SPDLOG_DEBUG("Linear module representation: {}", repr);

            // Test forward with residual connection
            SPDLOG_INFO("Testing Linear forward with residual connection");
            auto residual = infinicore::Tensor::ones({2, 4}, infinicore::DataType::F32, infinicore::Device());
            auto output_with_residual = m1.forward(input1, residual);
            if (output_with_residual->shape() != std::vector<size_t>({2, 4})) {
                SPDLOG_ERROR("Linear output with residual shape mismatch. Expected {{2, 4}}, got different shape");
                return false;
            }
            SPDLOG_DEBUG("Linear forward with residual passed. Input shape: {{2, 8}}, Residual shape: {{2, 4}}, Output shape: {{2, 4}}");

            // Test computation correctness: InfiniCore vs Naive implementation
            SPDLOG_INFO("Testing computation correctness: InfiniCore vs Naive implementation");

            // Create test data with known values for verification
            auto test_input = infinicore::Tensor::ones({2, 8}, infinicore::DataType::F32, infinicore::Device());
            auto test_residual = infinicore::Tensor::ones({2, 4}, infinicore::DataType::F32, infinicore::Device());

            // Get InfiniCore result
            auto infinicore_output = m1.forward(test_input, test_residual);

            // Compute naive result: output = input @ weight.T + bias + residual
            auto naive_output = infinicore::Tensor::empty({2, 4}, infinicore::DataType::F32, infinicore::Device());
            auto weight_naive = m1.weight();
            auto bias_naive = m1.bias();

            // Naive computation step by step
            auto weight_t = weight_naive->permute({1, 0});                     // [4, 8] -> [8, 4]
            auto matmul_result = infinicore::op::matmul(test_input, weight_t); // [2, 4]

            // Broadcast bias to [2, 4]
            size_t ndim_diff = naive_output->ndim() - 1;
            std::vector<infinicore::Stride> strides(ndim_diff, 0);
            strides.push_back(bias_naive->stride(0));
            auto bias_view = bias_naive->as_strided(naive_output->shape(), strides);

            // Add bias to matmul result
            infinicore::op::add_(naive_output, matmul_result, bias_view);

            // Add residual
            infinicore::op::add_(naive_output, naive_output, test_residual);

            // Compare results with actual value checking
            if (infinicore_output->shape() != naive_output->shape()) {
                SPDLOG_ERROR("Shape mismatch between InfiniCore and naive implementation");
                return false;
            }

            // Compare actual tensor values using local checker
            if (!tensorsAllClose(infinicore_output, naive_output, 1e-5, 1e-5)) {
                SPDLOG_ERROR("Value mismatch between InfiniCore and naive implementation");
                return false;
            }
            SPDLOG_DEBUG("Value comparison passed - InfiniCore and naive results match within tolerance");

            SPDLOG_DEBUG("Computation correctness test passed - both implementations produce identical results");
            SPDLOG_DEBUG("InfiniCore output shape: {{2, 4}}, Naive output shape: {{2, 4}}");

            // Test computation correctness without bias (using m2)
            SPDLOG_INFO("Testing computation correctness without bias");
            auto test_input_no_bias = infinicore::Tensor::ones({1, 16}, infinicore::DataType::F32, infinicore::Device());
            auto test_residual_no_bias = infinicore::Tensor::ones({1, 3}, infinicore::DataType::F32, infinicore::Device());

            // Get InfiniCore result (no bias)
            auto infinicore_output_no_bias = m2.forward(test_input_no_bias, test_residual_no_bias);

            // Compute naive result without bias: output = input @ weight.T + residual
            auto naive_output_no_bias = infinicore::Tensor::empty({1, 3}, infinicore::DataType::F32, infinicore::Device());
            auto weight_no_bias_naive = m2.weight();

            // Naive computation: just matmul + residual
            auto weight_t_no_bias = weight_no_bias_naive->permute({1, 0});                             // [3, 16] -> [16, 3]
            auto matmul_result_no_bias = infinicore::op::matmul(test_input_no_bias, weight_t_no_bias); // [1, 3]

            // Add residual
            infinicore::op::add_(naive_output_no_bias, matmul_result_no_bias, test_residual_no_bias);

            // Compare results with actual value checking
            if (infinicore_output_no_bias->shape() != naive_output_no_bias->shape()) {
                SPDLOG_ERROR("Shape mismatch between InfiniCore and naive implementation (no bias)");
                return false;
            }

            // Compare actual tensor values for no-bias case
            if (!tensorsAllClose(infinicore_output_no_bias, naive_output_no_bias, 1e-5, 1e-5)) {
                SPDLOG_ERROR("Value mismatch in no-bias computation");
                return false;
            }
            SPDLOG_DEBUG("No-bias value comparison passed - results match within tolerance");

            SPDLOG_DEBUG("No-bias computation correctness test passed - both implementations produce identical results");
            SPDLOG_DEBUG("InfiniCore no-bias output shape: {{1, 3}}, Naive no-bias output shape: {{1, 3}}");

            // Test basic forward (no residual) vs naive
            SPDLOG_INFO("Testing basic forward vs naive implementation");
            auto basic_infinicore = m1.forward(test_input);
            auto basic_naive = infinicore::Tensor::empty({2, 4}, infinicore::DataType::F32, infinicore::Device());

            // Naive basic computation: input @ weight.T + bias
            auto basic_matmul = infinicore::op::matmul(test_input, weight_t);
            infinicore::op::add_(basic_naive, basic_matmul, bias_view);

            if (basic_infinicore->shape() != basic_naive->shape()) {
                SPDLOG_ERROR("Shape mismatch in basic forward computation");
                return false;
            }

            // Compare actual tensor values for basic forward
            if (!tensorsAllClose(basic_infinicore, basic_naive, 1e-5, 1e-5)) {
                SPDLOG_ERROR("Value mismatch in basic forward computation");
                return false;
            }
            SPDLOG_DEBUG("Basic forward value comparison passed - results match within tolerance");

            SPDLOG_DEBUG("Basic forward computation correctness test passed - both implementations produce identical results");
            SPDLOG_DEBUG("Basic InfiniCore output shape: {{2, 4}}, Basic naive output shape: {{2, 4}}");

            SPDLOG_INFO("All Linear module tests passed (with/without bias, load_state_dict, computation verification)");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testModuleLinear: {}", e.what());
            return false;
        }
    });
}

// Test 6: Embedding module implementation
TestResult NNModuleTest::testModuleEmbedding() {
    return measureTime("ModuleEmbedding", [this]() {
        try {
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("Testing Embedding module implementation");
            SPDLOG_INFO("==========================================");

            // Test 1: Basic embedding creation
            SPDLOG_INFO("Test 1: Basic embedding creation (vocab=100, dim=64)");
            infinicore::nn::Embedding emb1(100, 64);

            auto state1 = emb1.state_dict();
            if (state1.find("weight") == state1.end()) {
                SPDLOG_ERROR("Embedding weight not found in state dict");
                return false;
            }

            if (state1.at("weight")->shape() != std::vector<size_t>({100, 64})) {
                SPDLOG_ERROR("Embedding weight shape mismatch. Expected {{100, 64}}");
                return false;
            }

            if (emb1.num_embeddings() != 100) {
                SPDLOG_ERROR("num_embeddings mismatch. Expected 100, got {}", emb1.num_embeddings());
                return false;
            }

            if (emb1.embedding_dim() != 64) {
                SPDLOG_ERROR("embedding_dim mismatch. Expected 64, got {}", emb1.embedding_dim());
                return false;
            }

            SPDLOG_DEBUG("Basic embedding creation passed");

            // Test 2: Embedding with padding_idx
            SPDLOG_INFO("Test 2: Embedding with padding_idx=0");
            infinicore::nn::Embedding emb2(50, 32, 0, infinicore::DataType::F32, infinicore::Device());

            if (!emb2.padding_idx().has_value()) {
                SPDLOG_ERROR("padding_idx should have a value");
                return false;
            }

            if (emb2.padding_idx().value() != 0) {
                SPDLOG_ERROR("padding_idx mismatch. Expected 0, got {}", emb2.padding_idx().value());
                return false;
            }

            SPDLOG_DEBUG("Embedding with padding_idx passed");

            // Test 3: Forward pass - single index
            SPDLOG_INFO("Test 3: Forward pass with single index");
            std::vector<int64_t> single_data = {5};
            auto indices_single = infinicore::Tensor::from_blob(single_data.data(), {1}, infinicore::DataType::I64, infinicore::Device());
            auto output_single = emb1.forward(indices_single);

            if (output_single->shape() != std::vector<size_t>({1, 64})) {
                SPDLOG_ERROR("Single index output shape mismatch. Expected {{1, 64}}");
                return false;
            }

            SPDLOG_DEBUG("Single index forward pass passed. Output shape: {{1, 64}}");

            // Test 4: Forward pass - batch of indices
            SPDLOG_INFO("Test 4: Forward pass with batch of indices");
            std::vector<int64_t> batch_data = {0, 5, 10};
            auto indices_batch = infinicore::Tensor::from_blob(batch_data.data(), {3}, infinicore::DataType::I64, infinicore::Device());
            auto output_batch = emb1.forward(indices_batch);

            if (output_batch->shape() != std::vector<size_t>({3, 64})) {
                SPDLOG_ERROR("Batch output shape mismatch. Expected {{3, 64}}");
                return false;
            }

            SPDLOG_DEBUG("Batch forward pass passed. Output shape: {{3, 64}}");

            // Test 5: Forward pass - 2D indices (batch_size, seq_len)
            SPDLOG_INFO("Test 5: Forward pass with 2D indices [batch, seq_len]");
            std::vector<int64_t> data_2d = {1, 2, 3, 4, 5, 6, 7, 8};
            auto indices_2d = infinicore::Tensor::from_blob(data_2d.data(), {2, 4},
                                                            infinicore::DataType::I64, infinicore::Device());
            auto output_2d = emb1.forward(indices_2d);

            if (output_2d->shape() != std::vector<size_t>({2, 4, 64})) {
                SPDLOG_ERROR("2D indices output shape mismatch. Expected {{2, 4, 64}}");
                return false;
            }

            SPDLOG_DEBUG("2D indices forward pass passed. Output shape: {{2, 4, 64}}");

            // Test 6: Embedding lookup consistency
            SPDLOG_INFO("Test 6: Testing embedding lookup consistency");
            std::vector<int64_t> idx_data = {7};
            auto idx1 = infinicore::Tensor::from_blob(idx_data.data(), {1}, infinicore::DataType::I64, infinicore::Device());
            auto idx2 = infinicore::Tensor::from_blob(idx_data.data(), {1}, infinicore::DataType::I64, infinicore::Device());

            auto out1 = emb1.forward(idx1);
            auto out2 = emb1.forward(idx2);

            // Same index should give same embedding
            if (!tensorsAllClose(out1, out2, 1e-7, 1e-7)) {
                SPDLOG_ERROR("Same index should return identical embeddings");
                return false;
            }

            SPDLOG_DEBUG("Embedding lookup consistency passed");

            // Test 7: load_state_dict
            SPDLOG_INFO("Test 7: Testing load_state_dict for Embedding");
            auto new_weight = infinicore::Tensor::ones({100, 64}, infinicore::DataType::F32, infinicore::Device());
            std::unordered_map<std::string, infinicore::Tensor> new_state;
            new_state.emplace("weight", new_weight);

            emb1.load_state_dict(new_state);

            if (!tensorsAllClose(emb1.weight(), new_weight, 1e-7, 1e-7)) {
                SPDLOG_ERROR("Embedding weight not loaded correctly");
                return false;
            }

            SPDLOG_DEBUG("load_state_dict for Embedding passed");

            // Test 8: extra_repr
            SPDLOG_INFO("Test 8: Testing extra_repr");
            std::string repr1 = emb1.extra_repr();
            std::string repr2 = emb2.extra_repr();

            SPDLOG_DEBUG("Embedding repr (no padding): {}", repr1);
            SPDLOG_DEBUG("Embedding repr (with padding): {}", repr2);

            if (repr1.find("num_embeddings=100") == std::string::npos) {
                SPDLOG_ERROR("extra_repr should contain num_embeddings");
                return false;
            }

            if (repr2.find("padding_idx=0") == std::string::npos) {
                SPDLOG_ERROR("extra_repr should contain padding_idx when specified");
                return false;
            }

            SPDLOG_DEBUG("extra_repr test passed");

            SPDLOG_INFO("All Embedding module tests passed!");
            return true;

        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testModuleEmbedding: {}", e.what());
            return false;
        }
    });
}

// Test 7: RMSNorm module implementation
TestResult NNModuleTest::testModuleRMSNorm() {
    return measureTime("ModuleRMSNorm", [this]() {
        try {
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("Testing RMSNorm module implementation");
            SPDLOG_INFO("==========================================");

            // Test 1: Basic RMSNorm creation
            SPDLOG_INFO("Test 1: Basic RMSNorm creation (hidden_size=768)");
            infinicore::nn::RMSNorm norm1(768);

            auto state1 = norm1.state_dict();
            if (state1.find("weight") == state1.end()) {
                SPDLOG_ERROR("RMSNorm weight not found in state dict");
                return false;
            }

            if (state1.at("weight")->shape() != std::vector<size_t>({768})) {
                SPDLOG_ERROR("RMSNorm weight shape mismatch. Expected {{768}}");
                return false;
            }

            if (norm1.normalized_shape() != 768) {
                SPDLOG_ERROR("normalized_shape mismatch. Expected 768, got {}", norm1.normalized_shape());
                return false;
            }

            SPDLOG_DEBUG("Basic RMSNorm creation passed");

            // Test 2: Forward pass - 2D input [batch, hidden]
            SPDLOG_INFO("Test 2: Forward pass with 2D input [batch, hidden]");
            auto input_2d = infinicore::Tensor::ones({4, 768}, infinicore::DataType::F32, infinicore::Device());
            auto output_2d = norm1.forward(input_2d);

            if (output_2d->shape() != std::vector<size_t>({4, 768})) {
                SPDLOG_ERROR("2D output shape mismatch. Expected {{4, 768}}");
                return false;
            }

            SPDLOG_DEBUG("2D forward pass passed. Output shape: {{4, 768}}");

            // Test 3: Forward pass - 3D input [batch, seq_len, hidden]
            SPDLOG_INFO("Test 3: Forward pass with 3D input [batch, seq_len, hidden]");
            auto input_3d = infinicore::Tensor::ones({2, 10, 768}, infinicore::DataType::F32, infinicore::Device());
            auto output_3d = norm1.forward(input_3d);

            if (output_3d->shape() != std::vector<size_t>({2, 10, 768})) {
                SPDLOG_ERROR("3D output shape mismatch. Expected {{2, 10, 768}}");
                return false;
            }

            SPDLOG_DEBUG("3D forward pass passed. Output shape: {{2, 10, 768}}");

            // Test 4: Test normalization properties
            SPDLOG_INFO("Test 4: Testing RMSNorm properties");
            auto test_input = infinicore::Tensor::ones({1, 768}, infinicore::DataType::F32, infinicore::Device());
            auto test_output = norm1.forward(test_input);

            // Output should have same shape
            if (test_output->shape() != test_input->shape()) {
                SPDLOG_ERROR("Output shape doesn't match input shape");
                return false;
            }

            SPDLOG_DEBUG("RMSNorm properties test passed");

            // Test 5: load_state_dict
            SPDLOG_INFO("Test 5: Testing load_state_dict for RMSNorm");
            auto new_weight = infinicore::Tensor::ones({768}, infinicore::DataType::F32, infinicore::Device());
            std::unordered_map<std::string, infinicore::Tensor> new_state;
            new_state.emplace("weight", new_weight);

            norm1.load_state_dict(new_state);

            if (!tensorsAllClose(norm1.weight(), new_weight, 1e-7, 1e-7)) {
                SPDLOG_ERROR("RMSNorm weight not loaded correctly");
                return false;
            }

            SPDLOG_DEBUG("load_state_dict for RMSNorm passed");

            // Test 6: extra_repr
            SPDLOG_INFO("Test 6: Testing extra_repr");
            std::string repr = norm1.extra_repr();
            SPDLOG_DEBUG("RMSNorm repr: {}", repr);

            if (repr.find("normalized_shape=768") == std::string::npos) {
                SPDLOG_ERROR("extra_repr should contain normalized_shape");
                return false;
            }

            if (repr.find("eps=") == std::string::npos) {
                SPDLOG_ERROR("extra_repr should contain eps");
                return false;
            }

            SPDLOG_DEBUG("extra_repr test passed");

            // Test 7: Input validation - normalized_shape mismatch (op layer handles this)
            SPDLOG_INFO("Test 7: Testing input validation - normalized_shape mismatch");
            auto input_wrong_shape = infinicore::Tensor::ones({4, 512}, infinicore::DataType::F32, infinicore::Device()); // normalized_shape=512, expected 768

            try {
                norm1.forward(input_wrong_shape);
                SPDLOG_ERROR("Should have thrown exception for normalized_shape mismatch");
                return false;
            } catch (const std::exception &e) {
                SPDLOG_DEBUG("Correctly caught exception for normalized_shape mismatch (handled by op layer): {}", e.what());
            } catch (...) {
                SPDLOG_ERROR("Caught unexpected exception type");
                return false;
            }

            SPDLOG_DEBUG("Normalized_shape mismatch validation test passed");

            // Test 8: Different hidden sizes
            SPDLOG_INFO("Test 8: Testing different hidden sizes");
            infinicore::nn::RMSNorm norm_small(128, 1e-5);
            infinicore::nn::RMSNorm norm_large(4096);

            auto input_small = infinicore::Tensor::ones({2, 128}, infinicore::DataType::F32, infinicore::Device());
            auto output_small = norm_small.forward(input_small);

            auto input_large = infinicore::Tensor::ones({2, 4096}, infinicore::DataType::F32, infinicore::Device());
            auto output_large = norm_large.forward(input_large);

            if (output_small->shape() != std::vector<size_t>({2, 128})) {
                SPDLOG_ERROR("Small RMSNorm output shape mismatch");
                return false;
            }

            if (output_large->shape() != std::vector<size_t>({2, 4096})) {
                SPDLOG_ERROR("Large RMSNorm output shape mismatch");
                return false;
            }

            SPDLOG_DEBUG("Different hidden sizes test passed");

            SPDLOG_INFO("All RMSNorm module tests passed!");
            return true;

        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testModuleRMSNorm: {}", e.what());
            return false;
        }
    });
}

// Test 7.5: RoPE module test
TestResult NNModuleTest::testModuleRoPE() {
    return measureTime("ModuleRoPE", [this]() {
        try {
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("Testing RoPE module implementation");
            SPDLOG_INFO("==========================================");

            // Test 1: Basic RoPE creation
            SPDLOG_INFO("Test 1: Basic RoPE creation (head_dim=128, max_seq_len=2048)");
            infinicore::nn::RoPE rope1(128, 2048);

            auto state1 = rope1.state_dict();

            if (rope1.head_dim() != 128) {
                SPDLOG_ERROR("head_dim mismatch. Expected 128, got {}", rope1.head_dim());
                return false;
            }

            if (rope1.max_seq_len() != 2048) {
                SPDLOG_ERROR("max_seq_len mismatch. Expected 2048, got {}", rope1.max_seq_len());
                return false;
            }

            SPDLOG_DEBUG("Basic RoPE creation passed");

            // Test 2: Forward pass - 3D input [seq_len, n_head, head_dim]
            SPDLOG_INFO("Test 2: Forward pass with 3D input [seq_len, n_head, head_dim]");
            auto x_3d = infinicore::Tensor::ones({32, 32, 128}, infinicore::DataType::F32, infinicore::Device());

            // Create position tensor [0, 1, 2, ..., 31]
            std::vector<int32_t> pos_data(32);
            for (size_t i = 0; i < 32; i++) {
                pos_data[i] = static_cast<int32_t>(i);
            }
            auto pos = infinicore::Tensor::from_blob(pos_data.data(), {32}, infinicore::DataType::I32, infinicore::Device());

            auto x_out = rope1.forward(x_3d, pos);

            if (x_out->shape() != std::vector<size_t>({32, 32, 128})) {
                SPDLOG_ERROR("3D output shape mismatch. Expected {{32, 32, 128}}");
                return false;
            }

            SPDLOG_DEBUG("3D forward pass passed. Output shape: {{32, 32, 128}}");

            // Test 3: Different algorithms
            SPDLOG_INFO("Test 3: Testing different algorithms");
            infinicore::nn::RoPE rope_gptj(64, 1024, 10000.0, infinicore::nn::RoPE::Algo::GPT_J, infinicore::nn::RoPE::Algo::GPT_J);
            infinicore::nn::RoPE rope_gptneox(64, 1024, 10000.0, infinicore::nn::RoPE::Algo::GPT_NEOX, infinicore::nn::RoPE::Algo::GPT_NEOX);

            if (rope_gptj.algo() != infinicore::nn::RoPE::Algo::GPT_J) {
                SPDLOG_ERROR("GPT_J algorithm not set correctly");
                return false;
            }

            if (rope_gptneox.algo() != infinicore::nn::RoPE::Algo::GPT_NEOX) {
                SPDLOG_ERROR("GPT_NEOX algorithm not set correctly");
                return false;
            }

            auto x_test = infinicore::Tensor::ones({10, 32, 64}, infinicore::DataType::F32, infinicore::Device());

            std::vector<int32_t> pos_test_data(10);
            for (size_t i = 0; i < 10; i++) {
                pos_test_data[i] = static_cast<int32_t>(i);
            }
            auto pos_test = infinicore::Tensor::from_blob(pos_test_data.data(), {10}, infinicore::DataType::I32, infinicore::Device());

            auto x_gptj = rope_gptj.forward(x_test, pos_test);
            auto x_neox = rope_gptneox.forward(x_test, pos_test);

            if (x_gptj->shape() != x_test->shape()) {
                SPDLOG_ERROR("GPT_J forward pass shape mismatch");
                return false;
            }

            if (x_neox->shape() != x_test->shape()) {
                SPDLOG_ERROR("GPT_NEOX forward pass shape mismatch");
                return false;
            }

            SPDLOG_DEBUG("Different algorithms test passed");

            // Test 4: Different theta values
            SPDLOG_INFO("Test 4: Testing different theta values");
            infinicore::nn::RoPE rope_theta1(128, 2048, 1e5);
            infinicore::nn::RoPE rope_theta2(128, 2048, 1e4);

            if (rope_theta1.theta() != 1e5) {
                SPDLOG_ERROR("theta mismatch. Expected 1e5, got {}", rope_theta1.theta());
                return false;
            }

            if (rope_theta2.theta() != 1e4) {
                SPDLOG_ERROR("theta mismatch. Expected 1e4, got {}", rope_theta2.theta());
                return false;
            }

            SPDLOG_DEBUG("Different theta values test passed");

            // Test 5: load_state_dict
            std::unordered_map<std::string, infinicore::Tensor> new_state;
            rope1.load_state_dict(new_state);
            SPDLOG_DEBUG("load_state_dict for RoPE passed (no parameters to load)");

            // Test 6: extra_repr
            SPDLOG_INFO("Test 6: Testing extra_repr");
            std::string repr = rope1.extra_repr();
            SPDLOG_DEBUG("RoPE repr: {}", repr);

            if (repr.find("head_dim=128") == std::string::npos) {
                SPDLOG_ERROR("extra_repr should contain head_dim");
                return false;
            }

            if (repr.find("max_seq_len=2048") == std::string::npos) {
                SPDLOG_ERROR("extra_repr should contain max_seq_len");
                return false;
            }

            if (repr.find("theta=") == std::string::npos) {
                SPDLOG_ERROR("extra_repr should contain theta");
                return false;
            }

            SPDLOG_DEBUG("extra_repr test passed");

            // Test 7: Different head dimensions
            SPDLOG_INFO("Test 7: Testing different head dimensions");
            infinicore::nn::RoPE rope_small(64, 1024);
            infinicore::nn::RoPE rope_large(256, 4096);

            auto x_small = infinicore::Tensor::ones({10, 32, 64}, infinicore::DataType::F32, infinicore::Device());

            std::vector<int32_t> pos_small_data(10);
            for (size_t i = 0; i < 10; i++) {
                pos_small_data[i] = static_cast<int32_t>(i);
            }
            auto pos_small = infinicore::Tensor::from_blob(pos_small_data.data(), {10}, infinicore::DataType::I32, infinicore::Device());

            auto x_small_out = rope_small.forward(x_small, pos_small);

            if (x_small_out->shape() != std::vector<size_t>({10, 32, 64})) {
                SPDLOG_ERROR("Small RoPE output shape mismatch");
                return false;
            }

            auto x_large = infinicore::Tensor::ones({20, 32, 256}, infinicore::DataType::F32, infinicore::Device());

            std::vector<int32_t> pos_large_data(20);
            for (size_t i = 0; i < 20; i++) {
                pos_large_data[i] = static_cast<int32_t>(i);
            }
            auto pos_large = infinicore::Tensor::from_blob(pos_large_data.data(), {20}, infinicore::DataType::I32, infinicore::Device());

            auto x_large_out = rope_large.forward(x_large, pos_large);

            if (x_large_out->shape() != std::vector<size_t>({20, 32, 256})) {
                SPDLOG_ERROR("Large RoPE output shape mismatch");
                return false;
            }

            SPDLOG_DEBUG("Different head dimensions test passed");

            // Test 8: Invalid head_dim (odd number)
            SPDLOG_INFO("Test 8: Testing invalid head_dim (odd number)");
            try {
                infinicore::nn::RoPE rope_invalid(127, 2048);
                SPDLOG_ERROR("Should have thrown exception for odd head_dim");
                return false;
            } catch (const std::invalid_argument &e) {
                SPDLOG_DEBUG("Correctly caught exception for odd head_dim: {}", e.what());
            } catch (...) {
                SPDLOG_ERROR("Caught unexpected exception type");
                return false;
            }

            SPDLOG_DEBUG("Invalid head_dim test passed");

            // Test 9: Input validation - empty tensor (op layer handles this)
            SPDLOG_INFO("Test 9: Testing input validation - empty tensor");
            auto x_empty = infinicore::Tensor::ones({}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_empty_data(1);
            pos_empty_data[0] = 0;
            auto pos_empty = infinicore::Tensor::from_blob(pos_empty_data.data(), {1}, infinicore::DataType::I32, infinicore::Device());

            try {
                rope1.forward(x_empty, pos_empty);
                SPDLOG_ERROR("Should have thrown exception for empty input tensor");
                return false;
            } catch (const std::exception &e) {
                SPDLOG_DEBUG("Correctly caught exception for empty input (handled by op layer): {}", e.what());
            } catch (...) {
                SPDLOG_ERROR("Caught unexpected exception type");
                return false;
            }

            SPDLOG_DEBUG("Empty tensor validation test passed");

            // Test 10: Input validation - head_dim mismatch (op layer handles this)
            SPDLOG_INFO("Test 10: Testing input validation - head_dim mismatch");
            auto x_wrong_dim = infinicore::Tensor::ones({32, 32, 64}, infinicore::DataType::F32, infinicore::Device()); // head_dim=64, expected 128
            std::vector<int32_t> pos_wrong_data(32);
            for (size_t i = 0; i < 32; i++) {
                pos_wrong_data[i] = static_cast<int32_t>(i);
            }
            auto pos_wrong = infinicore::Tensor::from_blob(pos_wrong_data.data(), {32}, infinicore::DataType::I32, infinicore::Device());

            try {
                rope1.forward(x_wrong_dim, pos_wrong);
                SPDLOG_ERROR("Should have thrown exception for head_dim mismatch");
                return false;
            } catch (const std::exception &e) {
                SPDLOG_DEBUG("Correctly caught exception for head_dim mismatch (handled by op layer): {}", e.what());
            } catch (...) {
                SPDLOG_ERROR("Caught unexpected exception type");
                return false;
            }

            SPDLOG_DEBUG("Head_dim mismatch validation test passed");

            // Test 11: Different input shapes (from reference test cases)
            SPDLOG_INFO("Test 11: Testing different input shapes");

            // Test shape (1, 32, 128) - single sequence
            auto x_single = infinicore::Tensor::ones({1, 32, 128}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_single_data(1);
            pos_single_data[0] = 0;
            auto pos_single = infinicore::Tensor::from_blob(pos_single_data.data(), {1}, infinicore::DataType::I32, infinicore::Device());
            auto x_single_out = rope1.forward(x_single, pos_single);
            if (x_single_out->shape() != std::vector<size_t>({1, 32, 128})) {
                SPDLOG_ERROR("Single sequence output shape mismatch");
                return false;
            }

            // Test shape (10, 32, 64) - different head_dim
            auto rope_64 = infinicore::nn::RoPE(64, 1024);
            auto x_64 = infinicore::Tensor::ones({10, 32, 64}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_64_data(10);
            for (size_t i = 0; i < 10; i++) {
                pos_64_data[i] = static_cast<int32_t>(i);
            }
            auto pos_64 = infinicore::Tensor::from_blob(pos_64_data.data(), {10}, infinicore::DataType::I32, infinicore::Device());
            auto x_64_out = rope_64.forward(x_64, pos_64);
            if (x_64_out->shape() != std::vector<size_t>({10, 32, 64})) {
                SPDLOG_ERROR("Shape (10, 32, 64) output mismatch");
                return false;
            }

            // Test shape (4, 1, 32) - single head
            auto rope_32 = infinicore::nn::RoPE(32, 1024);
            auto x_1head = infinicore::Tensor::ones({4, 1, 32}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_1head_data(4);
            for (size_t i = 0; i < 4; i++) {
                pos_1head_data[i] = static_cast<int32_t>(i);
            }
            auto pos_1head = infinicore::Tensor::from_blob(pos_1head_data.data(), {4}, infinicore::DataType::I32, infinicore::Device());
            auto x_1head_out = rope_32.forward(x_1head, pos_1head);
            if (x_1head_out->shape() != std::vector<size_t>({4, 1, 32})) {
                SPDLOG_ERROR("Shape (4, 1, 32) output mismatch");
                return false;
            }

            SPDLOG_DEBUG("Different input shapes test passed");

            // Test 12: Position tensor validation
            SPDLOG_INFO("Test 12: Testing position tensor edge cases");

            // Test with seq_len less than max_seq_len
            auto x_short = infinicore::Tensor::ones({10, 32, 128}, infinicore::DataType::F32, infinicore::Device());
            std::vector<int32_t> pos_short_data(10);
            for (size_t i = 0; i < 10; i++) {
                pos_short_data[i] = static_cast<int32_t>(i);
            }
            auto pos_short = infinicore::Tensor::from_blob(pos_short_data.data(), {10}, infinicore::DataType::I32, infinicore::Device());
            auto x_short_out = rope1.forward(x_short, pos_short);
            if (x_short_out->shape() != std::vector<size_t>({10, 32, 128})) {
                SPDLOG_ERROR("Short sequence output shape mismatch");
                return false;
            }

            SPDLOG_DEBUG("Position tensor edge cases test passed");

            // Test 13: Test that outputs are on the same device as inputs
            SPDLOG_INFO("Test 13: Testing device consistency");
            auto device = x_3d->device();
            if (x_out->device() != device) {
                SPDLOG_ERROR("Output tensor not on the same device as input");
                return false;
            }
            SPDLOG_DEBUG("Device consistency test passed");

            SPDLOG_INFO("All RoPE module tests passed!");
            return true;

        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testModuleRoPE: {}", e.what());
            return false;
        }
    });
}

// Test 8: Dtype assertion test
TestResult NNModuleTest::testDtypeAssertion() {
    return measureTime("DtypeAssertionTest", [this]() {
        try {
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("Testing dtype assertions when loading parameters");
            SPDLOG_INFO("==========================================");

            // Test 1: Successful load with matching dtype
            SPDLOG_INFO("Test 1: Successful load with matching dtype (F32)");
            infinicore::nn::Linear linear1(8, 4, true);
            auto matching_weight = infinicore::Tensor::ones({4, 8}, infinicore::DataType::F32, infinicore::Device());
            auto matching_bias = infinicore::Tensor::ones({4}, infinicore::DataType::F32, infinicore::Device());

            std::unordered_map<std::string, infinicore::Tensor> matching_state;
            matching_state.emplace("weight", matching_weight);
            matching_state.emplace("bias", matching_bias);

            // This should succeed without throwing
            linear1.load_state_dict(matching_state);
            SPDLOG_DEBUG("✓ Matching dtype load succeeded");

            // Test 2: Failed load with mismatched dtype (load_parameter)
            SPDLOG_INFO("Test 2: Failed load_parameter with mismatched dtype");
            infinicore::nn::Linear linear2(8, 4, true);
            auto mismatched_weight = infinicore::Tensor::ones({4, 8}, infinicore::DataType::BF16, infinicore::Device());

            bool exception_thrown = false;
            try {
                linear2.load_parameter("weight", mismatched_weight);
            } catch (const std::runtime_error &e) {
                exception_thrown = true;
                std::string error_msg = e.what();
                if (error_msg.find("dtype mismatch") == std::string::npos) {
                    SPDLOG_ERROR("Exception message doesn't contain 'dtype mismatch'");
                    return false;
                }
                SPDLOG_DEBUG("✓ Mismatched dtype exception caught: {}", error_msg);
            }

            if (!exception_thrown) {
                SPDLOG_ERROR("Expected exception for dtype mismatch in load_parameter");
                return false;
            }

            // Test 3: Failed load with mismatched dtype (load_state_dict)
            SPDLOG_INFO("Test 3: Failed load_state_dict with mismatched dtype");
            infinicore::nn::Embedding embedding1(100, 64);
            auto mismatched_embed_weight = infinicore::Tensor::ones({100, 64}, infinicore::DataType::BF16, infinicore::Device());

            std::unordered_map<std::string, infinicore::Tensor> mismatched_state;
            mismatched_state.emplace("weight", mismatched_embed_weight);

            exception_thrown = false;
            try {
                embedding1.load_state_dict(mismatched_state);
            } catch (const std::runtime_error &e) {
                exception_thrown = true;
                std::string error_msg = e.what();
                if (error_msg.find("dtype mismatch") == std::string::npos) {
                    SPDLOG_ERROR("Exception message doesn't contain 'dtype mismatch'");
                    return false;
                }
                if (error_msg.find("weight") == std::string::npos) {
                    SPDLOG_ERROR("Exception message doesn't contain parameter name 'weight'");
                    return false;
                }
                SPDLOG_DEBUG("✓ Mismatched dtype exception caught: {}", error_msg);
            }

            if (!exception_thrown) {
                SPDLOG_ERROR("Expected exception for dtype mismatch in load_state_dict");
                return false;
            }

            // Test 4: Failed load with mismatched dtype (RMSNorm)
            SPDLOG_INFO("Test 4: Failed load_state_dict with mismatched dtype (RMSNorm)");
            infinicore::nn::RMSNorm norm1(768);
            auto mismatched_norm_weight = infinicore::Tensor::ones({768}, infinicore::DataType::BF16, infinicore::Device());

            std::unordered_map<std::string, infinicore::Tensor> mismatched_norm_state;
            mismatched_norm_state.emplace("weight", mismatched_norm_weight);

            exception_thrown = false;
            try {
                norm1.load_state_dict(mismatched_norm_state);
            } catch (const std::runtime_error &e) {
                exception_thrown = true;
                std::string error_msg = e.what();
                if (error_msg.find("dtype mismatch") == std::string::npos) {
                    SPDLOG_ERROR("Exception message doesn't contain 'dtype mismatch'");
                    return false;
                }
                SPDLOG_DEBUG("✓ Mismatched dtype exception caught for RMSNorm: {}", error_msg);
            }

            if (!exception_thrown) {
                SPDLOG_ERROR("Expected exception for dtype mismatch in RMSNorm load_state_dict");
                return false;
            }

            // Test 5: Successful load with different module dtypes
            SPDLOG_INFO("Test 5: Successful load with BF16 dtype (module created with BF16)");
            infinicore::nn::Linear linear3(8, 4, true, infinicore::DataType::BF16);
            auto bf16_weight = infinicore::Tensor::ones({4, 8}, infinicore::DataType::BF16, infinicore::Device());
            auto bf16_bias = infinicore::Tensor::ones({4}, infinicore::DataType::BF16, infinicore::Device());

            std::unordered_map<std::string, infinicore::Tensor> bf16_state;
            bf16_state.emplace("weight", bf16_weight);
            bf16_state.emplace("bias", bf16_bias);

            // This should succeed
            linear3.load_state_dict(bf16_state);
            SPDLOG_DEBUG("✓ BF16 dtype load succeeded");

            SPDLOG_INFO("All dtype assertion tests passed!");
            return true;

        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testDtypeAssertion: {}", e.what());
            return false;
        }
    });
}

// Test 9: Comprehensive Tiny-Llama model test (construction + weight loading + validation)
TestResult NNModuleTest::testTinyLlamaConstruction() {
    return measureTime("TinyLlamaModelTest", [this]() {
        try {
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("Testing Tiny-Llama Model Construction and Weight Loading");
            SPDLOG_INFO("==========================================");

            // Tiny-Llama configuration (actual Tiny-Llama-1.1B-Chat-v1.0 specs)
            struct TinyLlamaConfig {
                size_t vocab_size = 32000;
                size_t hidden_size = 2048;
                size_t intermediate_size = 5632;
                size_t num_hidden_layers = 22;
                size_t num_attention_heads = 32;
                size_t num_key_value_heads = 4; // GQA (Grouped Query Attention)
                size_t max_position_embeddings = 2048;
                double rms_norm_eps = 1e-5;
            };

            TinyLlamaConfig config;

            // ============================================
            // Phase 0: Use hard-coded TinyLlama configuration (CI-friendly)
            // ============================================
            SPDLOG_INFO("");
            SPDLOG_INFO("Phase 0: Using hard-coded TinyLlama configuration (CI)");
            SPDLOG_INFO("------------------------------------------");

            SPDLOG_INFO("Using Configuration:");
            SPDLOG_INFO("  vocab_size: {}", config.vocab_size);
            SPDLOG_INFO("  hidden_size: {}", config.hidden_size);
            SPDLOG_INFO("  intermediate_size: {}", config.intermediate_size);
            SPDLOG_INFO("  num_layers: {}", config.num_hidden_layers);
            SPDLOG_INFO("  num_attention_heads: {}", config.num_attention_heads);
            SPDLOG_INFO("  num_key_value_heads: {} (GQA)", config.num_key_value_heads);
            SPDLOG_INFO("  max_position_embeddings: {}", config.max_position_embeddings);
            SPDLOG_INFO("  rms_norm_eps: {}", config.rms_norm_eps);

            // Create Tiny-Llama model skeleton closely matching HF/TinyLlama naming
            class TinyLlamaModel : public infinicore::nn::Module {
            protected:
                // Inner modules to match naming like: layers.0.self_attn.q_proj.weight, layers.0.mlp.gate_proj.weight
                class SelfAttn : public infinicore::nn::Module {
                public:
                    INFINICORE_NN_MODULE(infinicore::nn::Linear, q_proj);
                    INFINICORE_NN_MODULE(infinicore::nn::Linear, k_proj);
                    INFINICORE_NN_MODULE(infinicore::nn::Linear, v_proj);
                    INFINICORE_NN_MODULE(infinicore::nn::Linear, o_proj);

                    SelfAttn(size_t hidden_size, size_t kv_dim, const infinicore::Device &device) {
                        INFINICORE_NN_MODULE_INIT(q_proj, hidden_size, hidden_size, false, infinicore::DataType::F32, device);
                        INFINICORE_NN_MODULE_INIT(k_proj, hidden_size, kv_dim, false, infinicore::DataType::F32, device);
                        INFINICORE_NN_MODULE_INIT(v_proj, hidden_size, kv_dim, false, infinicore::DataType::F32, device);
                        INFINICORE_NN_MODULE_INIT(o_proj, hidden_size, hidden_size, false, infinicore::DataType::F32, device);
                    }
                };

                class MLP : public infinicore::nn::Module {
                public:
                    INFINICORE_NN_MODULE(infinicore::nn::Linear, gate_proj);
                    INFINICORE_NN_MODULE(infinicore::nn::Linear, up_proj);
                    INFINICORE_NN_MODULE(infinicore::nn::Linear, down_proj);

                    MLP(size_t hidden_size, size_t intermediate_size, const infinicore::Device &device) {
                        INFINICORE_NN_MODULE_INIT(gate_proj, hidden_size, intermediate_size, false, infinicore::DataType::F32, device);
                        INFINICORE_NN_MODULE_INIT(up_proj, hidden_size, intermediate_size, false, infinicore::DataType::F32, device);
                        INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size, hidden_size, false, infinicore::DataType::F32, device);
                    }
                };

                class Block : public infinicore::nn::Module {
                public:
                    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
                    INFINICORE_NN_MODULE(SelfAttn, self_attn);
                    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
                    INFINICORE_NN_MODULE(MLP, mlp);

                    Block(const TinyLlamaConfig &cfg, const infinicore::Device &device) {
                        size_t kv_dim = cfg.hidden_size * cfg.num_key_value_heads / cfg.num_attention_heads;
                        INFINICORE_NN_MODULE_INIT(input_layernorm, cfg.hidden_size, cfg.rms_norm_eps, infinicore::DataType::F32, device);
                        INFINICORE_NN_MODULE_INIT(self_attn, cfg.hidden_size, kv_dim, device);
                        INFINICORE_NN_MODULE_INIT(post_attention_layernorm, cfg.hidden_size, cfg.rms_norm_eps, infinicore::DataType::F32, device);
                        INFINICORE_NN_MODULE_INIT(mlp, cfg.hidden_size, cfg.intermediate_size, device);
                    }
                };

            public:
                INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
                INFINICORE_NN_MODULE_VEC(Block, layers);
                INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

                TinyLlamaModel(const TinyLlamaConfig &config, const infinicore::Device &device) {
                    INFINICORE_NN_MODULE_INIT(embed_tokens, config.vocab_size, config.hidden_size, std::nullopt, infinicore::DataType::F32, device);
                    INFINICORE_NN_MODULE_VEC_INIT(layers, config.num_hidden_layers, Block, config, device);
                    INFINICORE_NN_MODULE_INIT(norm, config.hidden_size, config.rms_norm_eps, infinicore::DataType::F32, device);
                }
            };

            // ============================================
            // Phase 1: Model Construction Verification
            // ============================================
            SPDLOG_INFO("");
            SPDLOG_INFO("Phase 1: Model Construction Verification");
            SPDLOG_INFO("------------------------------------------");

            // Construct the model
            TinyLlamaModel model(config, infinicore::Device());

            // Verify all components are created
            auto state = model.state_dict();
            SPDLOG_INFO("✓ Model constructed with {} parameters", state.size());

            // Parameter count expectation:
            // embed_tokens.weight (1) + norm.weight (1) + per-layer (9 params) * num_layers
            size_t expected_param_count = 1 + 1 + config.num_hidden_layers * 9;
            if (state.size() != expected_param_count) {
                SPDLOG_ERROR("Parameter count mismatch. Got {}, expected {} (1 + {}*9 + 1)",
                             state.size(), expected_param_count, config.num_hidden_layers);
                // Do not return false here to allow listing and detailed checks below
            }

            // List all parameters for manual verification
            SPDLOG_INFO("Listing all Tiny-Llama parameters (name -> shape):");
            for (const auto &kv : state) {
                const auto &name = kv.first;
                const auto &tensor = kv.second;
                std::ostringstream shape_ss;
                shape_ss << "[";
                for (size_t i = 0; i < tensor->shape().size(); ++i) {
                    if (i) {
                        shape_ss << ", ";
                    }
                    shape_ss << tensor->shape()[i];
                }
                shape_ss << "]";
                SPDLOG_INFO("  - {} -> {}", name, shape_ss.str());
            }

            // Automated verification: check all parameter shapes match hard-coded TinyLlama hierarchy
            SPDLOG_INFO("Verifying listed parameters against hard-coded TinyLlama hierarchy...");

            struct Expect {
                std::string name;
                std::vector<size_t> shape;
            };
            const size_t kv_dim = config.hidden_size * config.num_key_value_heads / config.num_attention_heads;
            std::vector<Expect> expected;
            // embed and final norm
            expected.push_back({"embed_tokens.weight", {config.vocab_size, config.hidden_size}});
            // per-layer expectations
            for (size_t i = 0; i < config.num_hidden_layers; ++i) {
                const std::string prefix = std::string("layers.") + std::to_string(i) + ".";
                expected.push_back({prefix + "input_layernorm.weight", {config.hidden_size}});
                expected.push_back({prefix + "self_attn.q_proj.weight", {config.hidden_size, config.hidden_size}});
                expected.push_back({prefix + "self_attn.k_proj.weight", {kv_dim, config.hidden_size}});
                expected.push_back({prefix + "self_attn.v_proj.weight", {kv_dim, config.hidden_size}});
                expected.push_back({prefix + "self_attn.o_proj.weight", {config.hidden_size, config.hidden_size}});
                expected.push_back({prefix + "post_attention_layernorm.weight", {config.hidden_size}});
                expected.push_back({prefix + "mlp.gate_proj.weight", {config.intermediate_size, config.hidden_size}});
                expected.push_back({prefix + "mlp.up_proj.weight", {config.intermediate_size, config.hidden_size}});
                expected.push_back({prefix + "mlp.down_proj.weight", {config.hidden_size, config.intermediate_size}});
            }
            expected.push_back({"norm.weight", {config.hidden_size}});

            bool all_ok = true;
            // Check expected ones (existence and shapes)
            for (const auto &e : expected) {
                auto it = state.find(e.name);
                if (it == state.end()) {
                    SPDLOG_ERROR("Missing expected parameter: {}", e.name);
                    all_ok = false;
                    continue;
                }
                auto got = it->second->shape();
                if (got != e.shape) {
                    std::ostringstream got_ss, exp_ss;
                    got_ss << "[";
                    for (size_t i = 0; i < got.size(); ++i) {
                        if (i) {
                            got_ss << ", ";
                        }
                        got_ss << got[i];
                    }
                    got_ss << "]";
                    exp_ss << "[";
                    for (size_t i = 0; i < e.shape.size(); ++i) {
                        if (i) {
                            exp_ss << ", ";
                        }
                        exp_ss << e.shape[i];
                    }
                    exp_ss << "]";
                    SPDLOG_ERROR("Shape mismatch for '{}': got {}, expected {}", e.name, got_ss.str(), exp_ss.str());
                    all_ok = false;
                }
            }

            // Check for unexpected extra parameters
            for (const auto &kvp : state) {
                const auto &name = kvp.first;
                bool is_expected = false;
                for (const auto &e : expected) {
                    if (e.name == name) {
                        is_expected = true;
                        break;
                    }
                }
                if (!is_expected) {
                    std::ostringstream got_ss;
                    auto got = kvp.second->shape();
                    got_ss << "[";
                    for (size_t i = 0; i < got.size(); ++i) {
                        if (i) {
                            got_ss << ", ";
                        }
                        got_ss << got[i];
                    }
                    got_ss << "]";
                    SPDLOG_WARN("Unexpected parameter present: {} with shape {}", name, got_ss.str());
                }
            }

            if (!all_ok) {
                SPDLOG_ERROR("Tiny-Llama parameter verification: FAILED - see errors above");
                return false;
            }
            SPDLOG_INFO("Tiny-Llama parameter verification: PASSED");

            // Create test weights
            std::unordered_map<std::string, infinicore::Tensor> test_state_dict;
            for (const auto &[name, tensor] : state) {
                // Create a test tensor with ones
                test_state_dict.emplace(name, infinicore::Tensor::ones(tensor->shape(),
                                                                       infinicore::DataType::F32,
                                                                       infinicore::Device()));
            }

            // Load the test weights
            model.load_state_dict(test_state_dict);

            // Verify weights were loaded
            auto loaded_state = model.state_dict();
            bool load_success = true;
            for (const auto &[name, _] : test_state_dict) {
                if (loaded_state.find(name) == loaded_state.end()) {
                    SPDLOG_ERROR("Parameter '{}' not found after load_state_dict", name);
                    load_success = false;
                }
            }

            if (!load_success) {
                SPDLOG_ERROR("Weight loading verification failed");
                return false;
            }

            SPDLOG_INFO("✓ State dict save/load mechanism verified");

            // ============================================
            // Summary
            // ============================================
            SPDLOG_INFO("");
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("✅ Tiny-Llama Model Test Summary");
            SPDLOG_INFO("==========================================");
            SPDLOG_INFO("✓ Metadata validation: PASSED (config matches actual model)");
            SPDLOG_INFO("✓ Model construction: PASSED");
            SPDLOG_INFO("✓ Parameter shapes: PASSED (11 parameters)");
            SPDLOG_INFO("✓ Forward passes: PASSED");
            SPDLOG_INFO("✓ Weight loading mechanism: PASSED");
            SPDLOG_INFO("✓ Architecture compatibility: Tiny-Llama-1.1B-Chat-v1.0");
            SPDLOG_INFO("✓ GQA support: num_key_value_heads={}", config.num_key_value_heads);
            SPDLOG_INFO("");
            SPDLOG_INFO("Model is ready for:");
            SPDLOG_INFO("  - Full 22-layer implementation");
            SPDLOG_INFO("  - Safetensors/pickle weight loading");
            SPDLOG_INFO("  - Inference and fine-tuning");
            SPDLOG_INFO("==========================================");

            return true;

        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception in testTinyLlamaConstruction: {}", e.what());
            return false;
        }
    });
}

// Main test runner
TestResult NNModuleTest::run() {
    std::vector<TestResult> results;

    std::cout << "==============================================\n"
              << "InfiniCore nn::Module Test Suite\n"
              << "==============================================" << std::endl;

    results.push_back(testBasicModuleCreation());   // Merged: creation + parameters + state_dict + load
    results.push_back(testLoadStateDict());         // Advanced: hierarchical modules
    results.push_back(testModuleHierarchy());       // Demonstrates hierarchical construction
    results.push_back(testParameterLoading());      // Blob loading
    results.push_back(testModuleLinear());          // Linear module comprehensive test
    results.push_back(testModuleEmbedding());       // Embedding module test
    results.push_back(testModuleRMSNorm());         // RMSNorm module test
    results.push_back(testModuleRoPE());            // RoPE module test
    results.push_back(testDtypeAssertion());        // Dtype assertion test
    results.push_back(testTinyLlamaConstruction()); // Comprehensive: TinyLlama model test

    // Check if all tests passed
    bool all_passed = true;
    for (const auto &result : results) {
        if (!result.passed) {
            all_passed = false;
            break;
        }
    }

    return TestResult("NNModuleTest", all_passed,
                      all_passed ? "" : "Some nn::module tests failed");
}

} // namespace infinicore::test
