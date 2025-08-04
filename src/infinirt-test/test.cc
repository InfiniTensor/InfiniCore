#include "test.h"
#include <algorithm>
#include <cstring>
#include <infinirt.h>
#include <iostream>
#include <numeric>
#include <vector>

bool testMemcpy(infiniDevice_t device, int deviceId, size_t dataSize) {

    std::cout << "==============================================\n"
              << "Testing memcpy on Device ID: " << deviceId << "\n"
              << "==============================================" << std::endl;

    // 分配主机内存
    std::cout << "[Device " << deviceId << "] Allocating host memory: " << dataSize * sizeof(float) << " bytes" << std::endl;
    std::vector<float> hostData(dataSize, 1.23f);
    std::vector<float> hostCopy(dataSize, 0.0f);

    // 分配设备内存
    void *deviceSrc = nullptr, *deviceDst = nullptr;
    size_t dataSizeInBytes = dataSize * sizeof(float);

    std::cout << "[Device " << deviceId << "] Allocating device memory: " << dataSizeInBytes << " bytes" << std::endl;
    if (infinirtMalloc(&deviceSrc, dataSizeInBytes) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to allocate device memory for deviceSrc." << std::endl;
        return false;
    }

    if (infinirtMalloc(&deviceDst, dataSizeInBytes) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to allocate device memory for deviceDst." << std::endl;
        infinirtFree(deviceSrc);
        return false;
    }

    // 复制数据到设备
    std::cout << "[Device " << deviceId << "] Copying data from host to device..." << std::endl;
    if (infinirtMemcpy(deviceSrc, hostData.data(), dataSizeInBytes, INFINIRT_MEMCPY_H2D) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to copy data from host to device." << std::endl;
        infinirtFree(deviceSrc);
        infinirtFree(deviceDst);
        return false;
    }

    // 设备内存间复制
    std::cout << "[Device " << deviceId << "] Copying data between device memory (D2D)..." << std::endl;
    if (infinirtMemcpy(deviceDst, deviceSrc, dataSizeInBytes, INFINIRT_MEMCPY_D2D) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to copy data from device to device." << std::endl;
        infinirtFree(deviceSrc);
        infinirtFree(deviceDst);
        return false;
    }

    // 设备数据复制回主机
    std::cout << "[Device " << deviceId << "] Copying data from device back to host..." << std::endl;
    if (infinirtMemcpy(hostCopy.data(), deviceDst, dataSizeInBytes, INFINIRT_MEMCPY_D2H) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to copy data from device to host." << std::endl;
        infinirtFree(deviceSrc);
        infinirtFree(deviceDst);
        return false;
    }

    // 数据验证
    std::cout << "[Device " << deviceId << "] Validating copied data..." << std::endl;
    if (std::memcmp(hostData.data(), hostCopy.data(), dataSizeInBytes) != 0) {
        std::cerr << "[Device " << deviceId << "] Data mismatch between hostData and hostCopy." << std::endl;
        infinirtFree(deviceSrc);
        infinirtFree(deviceDst);
        return false;
    }

    std::cout << "[Device " << deviceId << "] Data copied correctly!" << std::endl;

    // 释放设备内存
    std::cout << "[Device " << deviceId << "] Freeing device memory..." << std::endl;
    infinirtFree(deviceSrc);
    infinirtFree(deviceDst);

    std::cout << "[Device " << deviceId << "] Memory copy test PASSED!" << std::endl;

    return true;
}

bool testSetDevice(infiniDevice_t device, int deviceId) {

    std::cout << "Setting device " << device << " with ID: " << deviceId << std::endl;

    infiniStatus_t status = infinirtSetDevice(device, deviceId);

    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to set device " << device << " with ID " << deviceId << std::endl;
        return false;
    }

    return true;
}

bool testVirtualMem(infiniDevice_t device, int deviceId) {
    std::cout << "==============================================\n"
              << "Testing virtual memory on Device ID: " << deviceId << "\n"
              << "==============================================" << std::endl;

    infinirtMemProp_t prop;
    if (infinirtGetMemProp(&prop, device, deviceId) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to get memory property for device " << device << " with ID " << deviceId << std::endl;
        return false;
    }
    size_t min_granularity;
    if (infinirtGetMemGranularityMinimum(&min_granularity, prop) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to get memory granularity minimum for device " << device << " with ID " << deviceId << std::endl;
        return false;
    }
    std::cout << "Memory granularity minimum: " << min_granularity << " bytes" << std::endl;

    infinirtVirtualMemManager vm;
    if (infinirtCreateVirtualMemManager(&vm, device, 10 * min_granularity, 0) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to reserve virtual memory for device " << device << " with ID " << deviceId << std::endl;
        return false;
    }
    std::cout << "Virtual memory reserved: " << vm.len << " bytes" << std::endl;

    infinirtPhyMem phy_mem;
    if (infinirtCreatePhysicalMem(&phy_mem, min_granularity, prop) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to create physical memory for device " << device << " with ID " << deviceId << std::endl;
        return false;
    }
    std::cout << "Physical memory created: " << phy_mem.len << " bytes" << std::endl;

    void *mapped_ptr;
    if (infinirtMapVirtualMem(&mapped_ptr, &vm, min_granularity, &phy_mem) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to map virtual memory for device " << device << " with ID " << deviceId << std::endl;
        return false;
    }
    std::cout << "Virtual memory mapped at address: " << mapped_ptr << std::endl;

    size_t num_elements = min_granularity / sizeof(size_t);
    std::vector<size_t> host_data(num_elements);
    std::iota(host_data.begin(), host_data.end(), 0);

    if (infinirtMemcpy(mapped_ptr, host_data.data(), min_granularity, INFINIRT_MEMCPY_H2D) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to copy data from host to device." << std::endl;
        return false;
    }

    infinirtVirtualMemManager vm2;
    if (infinirtCreateVirtualMemManager(&vm2, device, 2 * min_granularity, 0) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to reserve second virtual memory for device " << device << " with ID " << deviceId << std::endl;
        return false;
    }

    void *mapped_ptr2;
    if (infinirtMapVirtualMem(&mapped_ptr2, &vm2, min_granularity, &phy_mem) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to map second virtual memory for device " << device << " with ID " << deviceId << std::endl;
        return false;
    }

    std::vector<size_t> host_data2(num_elements, 0);
    if (infinirtMemcpy(host_data2.data(), mapped_ptr2, min_granularity, INFINIRT_MEMCPY_D2H) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to copy data from device to host." << std::endl;
        return false;
    }

    if (!std::equal(host_data.begin(), host_data.end(), host_data2.begin())) {
        std::cerr << "Data mismatch between host_data and host_data2." << std::endl;
        return false;
    }

    std::cout << "Unmapping virtual memory..." << std::endl;
    if (infinirtUnmapVirtualMem(&vm, min_granularity) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to unmap virtual memory for device " << device << " with ID " << deviceId << std::endl;
        return false;
    }

    if (infinirtMemcpy(host_data.data(), mapped_ptr, min_granularity, INFINIRT_MEMCPY_D2H) == INFINI_STATUS_SUCCESS) {
        std::cerr << "Memory access after unmap should fail, but it succeeded." << std::endl;
        return false;
    }

    std::cout << "Virtual memory test PASSED!" << std::endl;

    return true;
}
