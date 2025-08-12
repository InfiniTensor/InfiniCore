#include "test.h"
#include <algorithm>
#include <cstring>
#include <infinirt.h>
#include <iostream>
#include <memory>
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
        void *vm;
        size_t vm_len = 10 * min_granularity;
        if (infinirtCreateVirtualMem(&vm, vm_len) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to reserve virtual memory" << std::endl;
            return false;
        }
        std::cout << "Virtual memory reserved: " << vm_len << " bytes" << std::endl;

        // Release virtual memory
        if (infinirtReleaseVirtualMem(vm, vm_len) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release virtual memory" << std::endl;
            return false;
        }
        std::cout << "Virtual memory released successfully" << std::endl;
    }

    // Test 2: Physical memory allocation and release
    {
        std::cout << "\nTest 2: Physical memory allocation and release" << std::endl;
        infinirtPhysicalMemoryHandle_t pm_handle;
        if (infinirtCreatePhysicalMem(&pm_handle, min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to create physical memory" << std::endl;
            return false;
        }
        std::cout << "Physical memory created: " << min_granularity << " bytes" << std::endl;

        // Release physical memory
        if (infinirtReleasePhysicalMem(pm_handle) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release physical memory" << std::endl;
            return false;
        }
        std::cout << "Physical memory released successfully" << std::endl;
    }

    // Test 3: Virtual memory mapping and unmapping with data verification
    {
        std::cout << "\nTest 3: Virtual memory mapping and data verification" << std::endl;

        // Create virtual memory regions
        void *vm1, *vm2;
        size_t vm_len = 10 * min_granularity;
        if (infinirtCreateVirtualMem(&vm1, vm_len) != INFINI_STATUS_SUCCESS || infinirtCreateVirtualMem(&vm2, 2 * min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to create virtual memory regions" << std::endl;
            return false;
        }

        // Create physical memory
        infinirtPhysicalMemoryHandle_t pm_handle;
        if (infinirtCreatePhysicalMem(&pm_handle, min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to create physical memory" << std::endl;
            return false;
        }

        // Map physical memory to both virtual memory regions
        if (infinirtMapVirtualMem(vm1, min_granularity, 0, pm_handle) != INFINI_STATUS_SUCCESS || infinirtMapVirtualMem(vm2, min_granularity, 0, pm_handle) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to map virtual memory" << std::endl;
            return false;
        }

        // Write data through first mapping
        size_t num_elements = min_granularity / sizeof(size_t);
        std::vector<size_t> host_data(num_elements);
        std::iota(host_data.begin(), host_data.end(), 0);
        if (infinirtMemcpy(vm1, host_data.data(), min_granularity, INFINIRT_MEMCPY_H2D) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to copy data to device" << std::endl;
            return false;
        }

        // Read data through second mapping
        std::vector<size_t> host_data2(num_elements, 0);
        if (infinirtMemcpy(host_data2.data(), vm2, min_granularity, INFINIRT_MEMCPY_D2H) != INFINI_STATUS_SUCCESS) {
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
        if (infinirtMemcpy(host_data.data(), vm1, min_granularity, INFINIRT_MEMCPY_D2H) == INFINI_STATUS_SUCCESS) {
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
        if (infinirtReleasePhysicalMem(pm_handle) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release physical memory" << std::endl;
            return false;
        }

        // Release virtual memory regions
        if (infinirtReleaseVirtualMem(vm1, vm_len) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release first virtual memory" << std::endl;
            return false;
        }
        if (infinirtReleaseVirtualMem(vm2, 2 * min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release second virtual memory" << std::endl;
            return false;
        }

        std::cout << "All resources cleaned up successfully" << std::endl;
    }

    // Test 4: Release virtual memory without unmapping
    {
        std::cout << "\nTest 4: Release virtual memory without unmapping" << std::endl;

        // Create virtual memory
        void *vm;
        size_t vm_len = 2 * min_granularity;
        if (infinirtCreateVirtualMem(&vm, vm_len) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to create virtual memory" << std::endl;
            return false;
        }

        // Create physical memory
        infinirtPhysicalMemoryHandle_t pm_handle;
        if (infinirtCreatePhysicalMem(&pm_handle, min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to create physical memory" << std::endl;
            infinirtReleaseVirtualMem(vm, vm_len);
            return false;
        }

        // Map virtual memory to physical memory
        if (infinirtMapVirtualMem(vm, min_granularity, 0, pm_handle) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to map virtual memory" << std::endl;
            infinirtReleasePhysicalMem(pm_handle);
            infinirtReleaseVirtualMem(vm, vm_len);
            return false;
        }

        std::cout << "Attempting to release virtual memory without unmapping first..." << std::endl;
        // Try to release virtual memory without unmapping - this should fail
        if (infinirtReleaseVirtualMem(vm, vm_len) == INFINI_STATUS_SUCCESS) {
            std::cerr << "ERROR: Virtual memory release succeeded without unmapping first!" << std::endl;
            // Clean up anyway
            infinirtUnmapVirtualMem(vm, min_granularity);
            infinirtReleasePhysicalMem(pm_handle);
            return false;
        }
        std::cout << "As expected, virtual memory release failed when mapped" << std::endl;

        // Clean up properly
        if (infinirtUnmapVirtualMem(vm, min_granularity) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to unmap virtual memory during cleanup" << std::endl;
            infinirtReleasePhysicalMem(pm_handle);
            return false;
        }

        if (infinirtReleasePhysicalMem(pm_handle) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release physical memory during cleanup" << std::endl;
            return false;
        }

        // Now release should succeed
        if (infinirtReleaseVirtualMem(vm, vm_len) != INFINI_STATUS_SUCCESS) {
            std::cerr << "Failed to release virtual memory after unmapping" << std::endl;
            return false;
        }
        std::cout << "Successfully released virtual memory after proper unmapping" << std::endl;
    }

    std::cout << "\nAll virtual memory tests PASSED!" << std::endl;
    return true;
}
