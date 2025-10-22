#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../context/allocators/device_caching_allocator.hpp"
#include "../context/allocators/device_pinned_allocator.hpp"
#include "../context/context_impl.hpp"
#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::memory {

inline void bind(py::module &m) {
    // Bind the Stat class
    py::class_<Stat>(m, "Stat")
        .def_readonly("current", &Stat::current)
        .def_readonly("peak", &Stat::peak)
        .def_readonly("allocated", &Stat::allocated)
        .def_readonly("freed", &Stat::freed)
        .def("increase", &Stat::increase)
        .def("decrease", &Stat::decrease)
        .def("reset_accumulated", &Stat::reset_accumulated)
        .def("reset_peak", &Stat::reset_peak);

    // Bind the StatType enum
    py::enum_<StatType>(m, "StatType")
        .value("AGGREGATE", StatType::AGGREGATE)
        .value("SMALL_POOL", StatType::SMALL_POOL)
        .value("LARGE_POOL", StatType::LARGE_POOL);

    // Bind the DeviceStats struct
    py::class_<DeviceStats>(m, "DeviceStats")
        .def_readonly("allocation", &DeviceStats::allocation)
        .def_readonly("segment", &DeviceStats::segment)
        .def_readonly("active", &DeviceStats::active)
        .def_readonly("inactive_split", &DeviceStats::inactive_split)
        .def_readonly("allocated_bytes", &DeviceStats::allocated_bytes)
        .def_readonly("reserved_bytes", &DeviceStats::reserved_bytes)
        .def_readonly("active_bytes", &DeviceStats::active_bytes)
        .def_readonly("inactive_split_bytes", &DeviceStats::inactive_split_bytes)
        .def_readonly("requested_bytes", &DeviceStats::requested_bytes)
        .def_readonly("num_alloc_retries", &DeviceStats::num_alloc_retries)
        .def_readonly("num_ooms", &DeviceStats::num_ooms)
        .def_readonly("oversize_allocations", &DeviceStats::oversize_allocations)
        .def_readonly("oversize_segments", &DeviceStats::oversize_segments)
        .def_readonly("num_sync_all_streams", &DeviceStats::num_sync_all_streams)
        .def_readonly("num_device_alloc", &DeviceStats::num_device_alloc)
        .def_readonly("num_device_free", &DeviceStats::num_device_free)
        .def_readonly("max_split_size", &DeviceStats::max_split_size);

    // Bind the Memory class
    py::class_<MemoryBlock, std::shared_ptr<MemoryBlock>>(m, "MemoryBlock")
        .def("data", &MemoryBlock::data)
        .def("device", &MemoryBlock::device)
        .def("size", &MemoryBlock::size)
        .def("is_pinned", &MemoryBlock::is_pinned);

    // Add functions to get memory statistics from the current runtime
    m.def(
        "get_device_memory_stats", []() -> DeviceStats {
            auto runtime = infinicore::ContextImpl::singleton().getCurrentRuntime();
            if (auto device_allocator = dynamic_cast<DeviceCachingAllocator *>(runtime->getDeviceMemoryAllocator())) {
                return device_allocator->getStats();
            }
            return DeviceStats{}; // Return empty stats if not a DeviceCachingAllocator
        },
        "Get device memory statistics from the current runtime");

    m.def(
        "get_pinned_host_memory_stats", []() -> DeviceStats {
            auto runtime = infinicore::ContextImpl::singleton().getCurrentRuntime();
            if (auto pinned_allocator = runtime->getPinnedHostMemoryAllocator()) {
                if (auto device_pinned_allocator = dynamic_cast<DevicePinnedHostAllocator *>(pinned_allocator)) {
                    return device_pinned_allocator->getStats();
                }
            }
            return DeviceStats{}; // Return empty stats if not available
        },
        "Get pinned host memory statistics from the current runtime");

    // Add functions to get memory statistics by device
    m.def(
        "get_device_memory_stats_by_device", [](const Device &device) -> DeviceStats {
            auto runtime = infinicore::ContextImpl::singleton().getRuntime(device);
            if (auto device_allocator = dynamic_cast<DeviceCachingAllocator *>(runtime->getDeviceMemoryAllocator())) {
                return device_allocator->getStats();
            }
            return DeviceStats{}; // Return empty stats if not a DeviceCachingAllocator
        },
        "Get device memory statistics for a specific device");

    m.def(
        "get_pinned_host_memory_stats_by_device", [](const Device &device) -> DeviceStats {
            auto runtime = infinicore::ContextImpl::singleton().getRuntime(device);
            if (auto pinned_allocator = runtime->getPinnedHostMemoryAllocator()) {
                if (auto device_pinned_allocator = dynamic_cast<DevicePinnedHostAllocator *>(pinned_allocator)) {
                    return device_pinned_allocator->getStats();
                }
            }
            return DeviceStats{}; // Return empty stats if not available
        },
        "Get pinned host memory statistics for a specific device");
}

} // namespace infinicore::memory
