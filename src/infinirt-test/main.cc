#include "test.h"
#include <infinirt.h>

struct ParsedArgs {
    infiniDevice_t device_type = INFINI_DEVICE_CPU; // 默认 CPU
};

void printUsage() {
    std::cout << "Usage:\n"
              << "  infinirt-test [--<device>]\n\n"
              << "  --<device>   Specify device type.\n"
              << "               Available devices: cpu, nvidia, cambricon, ascend,\n"
              << "               metax, moore, iluvatar, kunlun, sugon\n"
              << "               Default is CPU.\n"
              << std::endl;
    exit(EXIT_FAILURE);
}

#define PARSE_DEVICE(FLAG, DEVICE) \
    else if (arg == FLAG) {        \
        args.device_type = DEVICE; \
    }

ParsedArgs parseArgs(int argc, char *argv[]) {
    ParsedArgs args;

    if (argc < 2) {
        return args; // 默认使用 CPU
    }

    std::string arg = argv[1];
    if (arg == "--help" || arg == "-h") {
        printUsage();
    }

    try {
        if (arg == "--cpu") {
            args.device_type = INFINI_DEVICE_CPU;
        }
        PARSE_DEVICE("--nvidia", INFINI_DEVICE_NVIDIA)
        PARSE_DEVICE("--cambricon", INFINI_DEVICE_CAMBRICON)
        PARSE_DEVICE("--ascend", INFINI_DEVICE_ASCEND)
        PARSE_DEVICE("--metax", INFINI_DEVICE_METAX)
        PARSE_DEVICE("--moore", INFINI_DEVICE_MOORE)
        PARSE_DEVICE("--iluvatar", INFINI_DEVICE_ILUVATAR)
        PARSE_DEVICE("--kunlun", INFINI_DEVICE_KUNLUN)
        PARSE_DEVICE("--sugon", INFINI_DEVICE_SUGON)
        else {
            printUsage();
        }
    } catch (const std::exception &) {
        printUsage();
    }

    return args;
}
int main(int argc, char *argv[]) {

    ParsedArgs args = parseArgs(argc, argv);
    std::cout << "Testing Device: " << args.device_type << std::endl;
    infiniDevice_t device = args.device_type;

    // 获取设备总数
    std::vector<int> deviceCounts(INFINI_DEVICE_TYPE_COUNT, 0);
    if (infinirtGetAllDeviceCount(deviceCounts.data()) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to get total device count." << std::endl;
        return 1;
    }

    int numDevices = deviceCounts[device];
    std::cout << "Device Type: " << device << " | Available Devices: " << numDevices << std::endl;

    if (numDevices == 0) {
        std::cerr << "Device type " << device << " has no available devices." << std::endl;
        return 1;
    }

    for (int deviceId = 0; deviceId < numDevices; ++deviceId) {
        if (!test_setDevice(device, deviceId)) {
            return 1;
        }
        // test_memcpy
        size_t dataSize = 1024;
        if (!test_memcpy(device, deviceId, dataSize)) {
            return 1;
        }
    }

    return 0;
}
