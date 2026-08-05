#include "infiniccl_test.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

enum class TestFunc {
    AllReduce,
    Broadcast,
    Send,
    Recv,
};

struct ParsedArgs {
    infiniDevice_t device_type = INFINI_DEVICE_CPU;
    bool parsed_device = false;
    bool multi_node = false;
    TestFunc func = TestFunc::AllReduce;
    bool has_master_addr = false;
    std::string master_addr;
    int master_port = 29500;
};

void printUsage() {
    std::cout << "Usage:" << std::endl
              << std::endl;
    std::cout << "infiniccl-test --<device> [--func allreduce|broadcast|send|recv]" << std::endl
              << "infiniccl-test --<device> --multi-node --func send|recv "
              << "[--master-addr <ip>] [--master-port <port>]" << std::endl
              << std::endl;
    std::cout << "  --<device>" << std::endl;
    std::cout << "    Specify the device type --(nvidia|cambricon|ascend|metax|moore|iluvatar|qy|kunlun|hygon|ali)." << std::endl
              << std::endl;
    std::cout << "  --func" << std::endl
              << "    allreduce: local all-reduce test across all visible devices." << std::endl
              << "    broadcast: local broadcast test from device 0 to all visible devices." << std::endl
              << "    send: local device 0 sends to device 1, or multi-node sender on visible device 0." << std::endl
              << "    recv: local device 1 sends to device 0, or multi-node receiver on visible device 0." << std::endl
              << std::endl;
    std::cout << "For multi-node send/recv, the process without --master-addr is the TCP master."
              << std::endl;
    exit(-1);
}

#define PARSE_DEVICE(FLAG, DEVICE) \
    if (arg == FLAG) {             \
        args.device_type = DEVICE; \
        args.parsed_device = true; \
    }

ParsedArgs parseArgs(int argc, char *argv[]) {
    if (argc < 2) {
        printUsage();
    }

    ParsedArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
        }
        // clang-format off
        PARSE_DEVICE("--nvidia", INFINI_DEVICE_NVIDIA)
        else PARSE_DEVICE("--cambricon", INFINI_DEVICE_CAMBRICON)
        else PARSE_DEVICE("--ascend", INFINI_DEVICE_ASCEND)
        else PARSE_DEVICE("--metax", INFINI_DEVICE_METAX)
        else PARSE_DEVICE("--moore", INFINI_DEVICE_MOORE)
        else PARSE_DEVICE("--iluvatar", INFINI_DEVICE_ILUVATAR)
        else PARSE_DEVICE("--qy", INFINI_DEVICE_QY)
        else PARSE_DEVICE("--kunlun", INFINI_DEVICE_KUNLUN)
        else PARSE_DEVICE("--hygon", INFINI_DEVICE_HYGON)
        else PARSE_DEVICE("--ali", INFINI_DEVICE_ALI)
        else if (arg == "--multi-node") {
            args.multi_node = true;
        } else if (arg == "--func" && i + 1 < argc) {
            std::string func = argv[++i];
            if (func == "allreduce") {
                args.func = TestFunc::AllReduce;
            } else if (func == "broadcast") {
                args.func = TestFunc::Broadcast;
            } else if (func == "send") {
                args.func = TestFunc::Send;
            } else if (func == "recv") {
                args.func = TestFunc::Recv;
            } else {
                printUsage();
            }
        } else if (arg == "--master-addr" && i + 1 < argc) {
            args.has_master_addr = true;
            args.master_addr = argv[++i];
        } else if (arg == "--master-port" && i + 1 < argc) {
            args.master_port = std::atoi(argv[++i]);
        } else {
            printUsage();
        }
        // clang-format on
    }

    if (!args.parsed_device || args.master_port <= 0 || args.master_port > 65535) {
        printUsage();
    }
    if (args.multi_node && (args.func == TestFunc::AllReduce || args.func == TestFunc::Broadcast)) {
        printUsage();
    }
    if (!args.multi_node && args.has_master_addr) {
        printUsage();
    }
    return args;
}

int main(int argc, char *argv[]) {
    ParsedArgs args = parseArgs(argc, argv);
    infinirtInit();

    if (args.multi_node) {
        return testMultiNodeSend(args.device_type, args.func == TestFunc::Send,
                                 args.has_master_addr ? args.master_addr.c_str() : nullptr,
                                 args.master_port);
    }

    int ndevice = 0;
    if (infinirtGetDeviceCount(args.device_type, &ndevice) != INFINI_STATUS_SUCCESS) {
        std::cout << "Failed to get device count" << std::endl;
        return -1;
    }
    if (ndevice == 0) {
        std::cout << "No devices found. Tests skipped." << std::endl;
        return 0;
    }

    std::cout << "Found " << ndevice << " devices. Running tests..." << std::endl;
    if (args.func == TestFunc::Send) {
        return testSend(args.device_type, ndevice, true);
    }
    if (args.func == TestFunc::Recv) {
        return testSend(args.device_type, ndevice, false);
    }
    if (args.func == TestFunc::Broadcast) {
        return testBroadcast(args.device_type, ndevice);
    }
    return testAllReduce(args.device_type, ndevice);
}
