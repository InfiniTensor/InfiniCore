#include "memory_test.h"
#include "tensor/test_d2h_issue.h"
#include "tensor/test_tensor_destructor.h"
#include "tensor/test_tensor_usage.h"
#include "test_nn_module.h"
#include "test_runner.h"
#include <iostream>
#include <memory>
#include <spdlog/spdlog.h>
#include <vector>

struct ParsedArgs {
    infiniDevice_t device_type = INFINI_DEVICE_CPU;
    bool run_basic = true;
    bool run_tensor = true;
    bool run_concurrency = true;
    bool run_exception_safety = true;
    bool run_memory_leak = true;
    bool run_performance = true;
    bool run_stress = true;
    bool run_module = false;
    int num_threads = 4;
    int iterations = 1000;
};

void printUsage() {
    std::cout << "Usage:" << std::endl
              << "  infinicore-test [--<device>] [--test <test_name>] [--threads <num>] [--iterations <num>]" << std::endl
              << std::endl
              << "Options:" << std::endl
              << "  --<device>        Specify the device type (default: cpu)" << std::endl
              << "  --test <name>     Run specific test (basic|tensor|concurrency|exception|leak|performance|stress|module|all)" << std::endl
              << "  --threads <num>   Number of threads for concurrency tests (default: 4)" << std::endl
              << "  --iterations <num> Number of iterations for stress tests (default: 1000)" << std::endl
              << "  --help            Show this help message" << std::endl
              << std::endl
              << "Available devices:" << std::endl
              << "  cpu         - Default" << std::endl
              << "  nvidia" << std::endl
              << "  cambricon" << std::endl
              << "  ascend" << std::endl
              << "  metax" << std::endl
              << "  moore" << std::endl
              << "  iluvatar" << std::endl
              << "  qy" << std::endl
              << "  kunlun" << std::endl
              << "  hygon" << std::endl
              << std::endl
              << "Available tests:" << std::endl
              << "  basic       - Basic memory allocation and deallocation tests" << std::endl
              << "  tensor      - Tensor-related tests (destructor, usage)" << std::endl
              << "  concurrency - Thread safety and concurrent access tests" << std::endl
              << "  exception   - Exception safety tests" << std::endl
              << "  leak        - Memory leak detection tests" << std::endl
              << "  performance - Performance and benchmark tests" << std::endl
              << "  stress      - Stress tests with high load" << std::endl
              << "  module      - Neural network module tests" << std::endl
              << "  all         - Run all tests (default)" << std::endl
              << std::endl;
    exit(EXIT_SUCCESS);
}

ParsedArgs parseArgs(int argc, char *argv[]) {
    ParsedArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage();
        } else if (arg == "--cpu") {
            args.device_type = INFINI_DEVICE_CPU;
        } else if (arg == "--nvidia") {
            args.device_type = INFINI_DEVICE_NVIDIA;
        } else if (arg == "--cambricon") {
            args.device_type = INFINI_DEVICE_CAMBRICON;
        } else if (arg == "--ascend") {
            args.device_type = INFINI_DEVICE_ASCEND;
        } else if (arg == "--metax") {
            args.device_type = INFINI_DEVICE_METAX;
        } else if (arg == "--moore") {
            args.device_type = INFINI_DEVICE_MOORE;
        } else if (arg == "--iluvatar") {
            args.device_type = INFINI_DEVICE_ILUVATAR;
        } else if (arg == "--qy") {
            args.device_type = INFINI_DEVICE_QY;
        } else if (arg == "--kunlun") {
            args.device_type = INFINI_DEVICE_KUNLUN;
        } else if (arg == "--hygon") {
            args.device_type = INFINI_DEVICE_HYGON;
        } else if (arg == "--test") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --test requires a test name" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::string test_name = argv[++i];
            args.run_basic = args.run_tensor = args.run_concurrency = args.run_exception_safety = args.run_memory_leak = args.run_performance = args.run_stress = args.run_module = false;

            if (test_name == "basic") {
                args.run_basic = true;
            } else if (test_name == "tensor") {
                args.run_tensor = true;
            } else if (test_name == "concurrency") {
                args.run_concurrency = true;
            } else if (test_name == "exception") {
                args.run_exception_safety = true;
            } else if (test_name == "leak") {
                args.run_memory_leak = true;
            } else if (test_name == "performance") {
                args.run_performance = true;
            } else if (test_name == "stress") {
                args.run_stress = true;
            } else if (test_name == "module") {
                args.run_module = true;
            } else if (test_name == "all") {
                args.run_basic = args.run_tensor = args.run_concurrency = args.run_exception_safety = args.run_memory_leak = args.run_performance = args.run_module = true;
            } else {
                std::cerr << "Error: Unknown test name: " << test_name << std::endl;
                exit(EXIT_FAILURE);
            }
        } else if (arg == "--threads") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --threads requires a number" << std::endl;
                exit(EXIT_FAILURE);
            }
            args.num_threads = std::stoi(argv[++i]);
            if (args.num_threads <= 0) {
                std::cerr << "Error: Number of threads must be positive" << std::endl;
                exit(EXIT_FAILURE);
            }
        } else if (arg == "--iterations") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --iterations requires a number" << std::endl;
                exit(EXIT_FAILURE);
            }
            args.iterations = std::stoi(argv[++i]);
            if (args.iterations <= 0) {
                std::cerr << "Error: Number of iterations must be positive" << std::endl;
                exit(EXIT_FAILURE);
            }
        } else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

int main(int argc, char *argv[]) {
    try {
        ParsedArgs args = parseArgs(argc, argv);
        SPDLOG_INFO("Arguments parsed successfully");

        std::cout << "==============================================\n"
                  << "InfiniCore Memory Management Test Suite\n"
                  << "==============================================\n"
                  << "Device: " << static_cast<int>(args.device_type) << "\n"
                  << "Threads: " << args.num_threads << "\n"
                  << "Iterations: " << args.iterations << "\n"
                  << "==============================================" << std::endl;

        SPDLOG_INFO("About to initialize InfiniCore context");
        // Initialize InfiniCore context
        infinicore::context::setDevice(infinicore::Device(static_cast<infinicore::Device::Type>(args.device_type), 0));
        SPDLOG_INFO("InfiniCore context initialized successfully");

        SPDLOG_INFO("Creating test runner");
        // Create test runner
        infinicore::test::InfiniCoreTestRunner runner;
        SPDLOG_INFO("Test runner created successfully");

        // Add tests based on arguments
        if (args.run_basic) {
            runner.addTest(std::make_unique<infinicore::test::BasicMemoryTest>());

            // Add device switch test to basic tests (critical regression test)
            runner.addTest(std::make_unique<infinicore::test::DeviceSwitchTest>());
        }

        if (args.run_tensor) {
            runner.addTest(std::make_unique<infinicore::test::TensorDestructorTest>());

            // Add tensor usage test (tests operations used in InfiniLM weight loading)
            runner.addTest(std::make_unique<infinicore::test::TensorUsageTest>());

            // Add D2H issue test (reproduces the segfault issue)
            runner.addTest(std::make_unique<infinicore::test::D2HIssueTest>());
        }

        if (args.run_module) {
            runner.addTest(std::make_unique<infinicore::test::NNModuleTest>());
        }

        if (args.run_concurrency) {
            runner.addTest(std::make_unique<infinicore::test::ConcurrencyTest>());
        }

        if (args.run_exception_safety) {
            // runner.addTest(std::make_unique<infinicore::test::ExceptionSafetyTest>());
        }

        if (args.run_memory_leak) {
            runner.addTest(std::make_unique<infinicore::test::MemoryLeakTest>());
        }

        if (args.run_performance) {
            // TODO: Segmentation fault when LOG LEVEL is set to INFO, passed when set to DEBUG
            // runner.addTest(std::make_unique<infinicore::test::PerformanceTest>());
        }

        if (args.run_stress) {
            runner.addTest(std::make_unique<infinicore::test::StressTest>());
        }

        SPDLOG_INFO("About to run all tests");
        // Run all tests
        auto results = runner.runAllTests();
        SPDLOG_INFO("All tests completed");

        // Count results and collect failed tests
        size_t passed = 0, failed = 0;
        std::vector<infinicore::test::TestResult> failed_tests;
        for (const auto &result : results) {
            if (result.passed) {
                passed++;
            } else {
                failed++;
                failed_tests.push_back(result);
            }
        }

        // Print list of failed tests if any
        if (!failed_tests.empty()) {
            std::cout << "\n==============================================\n"
                      << "❌ FAILED TESTS\n"
                      << "==============================================" << std::endl;
            for (const auto &test : failed_tests) {
                std::cout << "  • " << test.test_name;
                if (!test.error_message.empty()) {
                    std::cout << "\n    Error: " << test.error_message;
                }
                std::cout << "\n    Duration: " << test.duration.count() << "μs" << std::endl;
            }
        }

        // Print final summary
        std::cout << "\n==============================================\n"
                  << "Final Results\n"
                  << "==============================================\n"
                  << "Total Tests: " << results.size() << "\n"
                  << "Passed: " << passed << "\n"
                  << "Failed: " << failed << "\n"
                  << "==============================================" << std::endl;

        // Exit with appropriate code
        if (failed > 0) {
            std::cout << "\n❌ Some tests failed. Please review the failed tests list above." << std::endl;
            return EXIT_FAILURE;
        } else {
            std::cout << "\n✅ All tests passed!" << std::endl;
            return EXIT_SUCCESS;
        }

    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Fatal error: Unknown exception" << std::endl;
        return EXIT_FAILURE;
    }
}
