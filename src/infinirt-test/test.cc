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

    // Get minimum granularity
    size_t min_granularity;
    if (infinirtGetMemGranularityMinimum(&min_granularity) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to get memory granularity minimum" << std::endl;
        return false;
    }
    std::cout << "Memory granularity minimum: " << min_granularity << " bytes" << std::endl;

    // Test 1: Basic virtual memory allocation and release
    {
        std::cout << "\nTest 1: Basic virtual memory allocation and release" << std::endl;
        infinirtVirtualMem_t vm;
        size_t vm_len = 10 * min_granularity;
        if (infinirtCreateVirtualMem(&vm, vm_len) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to reserve virtual memory" << std::endl;
            return false;
        }
        std::cout << "Virtual memory reserved: " << vm_len << " bytes" << std::endl;

        // Release virtual memory
        if (infinirtReleaseVirtualMem(vm) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release virtual memory" << std::endl;
            return false;
        }
        std::cout << "Virtual memory released successfully" << std::endl;
    }

    // Test 2: Physical memory allocation and release
    {
        std::cout << "\nTest 2: Physical memory allocation and release" << std::endl;
        infinirtPhyMem_t phy_mem;
        if (infinirtCreatePhysicalMem(&phy_mem, min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to create physical memory" << std::endl;
            return false;
        }
        std::cout << "Physical memory created: " << min_granularity << " bytes" << std::endl;

        // Release physical memory
        if (infinirtReleasePhysicalMem(phy_mem) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release physical memory" << std::endl;
            return false;
        }
        std::cout << "Physical memory released successfully" << std::endl;
    }

    // Test 3: Virtual memory mapping and unmapping with data verification
    {
        std::cout << "\nTest 3: Virtual memory mapping and data verification" << std::endl;

        // Create virtual memory regions
        infinirtVirtualMem_t vm1, vm2;
        size_t vm_len = 10 * min_granularity;
        if (infinirtCreateVirtualMem(&vm1, vm_len) != INFINI_STATUS_SUCCESS || infinirtCreateVirtualMem(&vm2, 2 * min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to create virtual memory regions" << std::endl;
            return false;
        }

        // Create physical memory
        infinirtPhyMem_t phy_mem;
        if (infinirtCreatePhysicalMem(&phy_mem, min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to create physical memory" << std::endl;
            return false;
        }

        // Map physical memory to both virtual memory regions
        void *mapped_ptr1, *mapped_ptr2;
        if (infinirtMapVirtualMem(&mapped_ptr1, vm1, min_granularity, phy_mem) != INFINI_STATUS_SUCCESS || infinirtMapVirtualMem(&mapped_ptr2, vm2, min_granularity, phy_mem) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to map virtual memory" << std::endl;
            return false;
        }

        // Write data through first mapping
        size_t num_elements = min_granularity / sizeof(size_t);
        std::vector<size_t> host_data(num_elements);
        std::iota(host_data.begin(), host_data.end(), 0);
        if (infinirtMemcpy(mapped_ptr1, host_data.data(), min_granularity, INFINIRT_MEMCPY_H2D) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to copy data to device" << std::endl;
            return false;
        }

        // Read data through second mapping
        std::vector<size_t> host_data2(num_elements, 0);
        if (infinirtMemcpy(host_data2.data(), mapped_ptr2, min_granularity, INFINIRT_MEMCPY_D2H) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to copy data from device" << std::endl;
            return false;
        }

        // Verify data
        if (!std::equal(host_data.begin(), host_data.end(), host_data2.begin())) {
            std::cerr << "Data mismatch between mappings" << std::endl;
            return false;
        }

        // Test unmapping
        if (infinirtUnmapVirtualMem(vm1, min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to unmap virtual memory" << std::endl;
            return false;
        }

        // Verify memory access fails after unmapping
        if (infinirtMemcpy(host_data.data(), mapped_ptr1, min_granularity, INFINIRT_MEMCPY_D2H) == INFINI_STATUS_SUCCESS) {
            std::cerr << "Memory access after unmap should fail" << std::endl;
            return false;
        }

        // Clean up all resources
        std::cout << "\nCleaning up resources..." << std::endl;

        // Unmap remaining mapping
        if (infinirtUnmapVirtualMem(vm2, min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to unmap second virtual memory" << std::endl;
            return false;
        }

        // Release physical memory
        if (infinirtReleasePhysicalMem(phy_mem) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release physical memory" << std::endl;
            return false;
        }

        // Release virtual memory regions
        if (infinirtReleaseVirtualMem(vm1) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release first virtual memory" << std::endl;
            return false;
        }
        if (infinirtReleaseVirtualMem(vm2) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release second virtual memory" << std::endl;
            return false;
        }

        std::cout << "All resources cleaned up successfully" << std::endl;
    }

    std::cout << "\nAll virtual memory tests PASSED!" << std::endl;
    return true;
}
