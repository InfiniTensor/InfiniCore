#pragma once

#include "../device.hpp"
#include "../memory.hpp"

#include <infiniop.h>
#include <infinirt.h>

#include <memory>

namespace infinicore {

namespace context {
void setDevice(Device device);
Device getDevice();
size_t getDeviceCount(Device::Type type);

infinirtStream_t getStream();
infiniopHandle_t getInfiniopHandle();

void syncStream();
void syncDevice();

std::shared_ptr<Memory> allocateMemory(size_t size);
std::shared_ptr<Memory> allocateHostMemory(size_t size);
std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);

void memcpyH2D(void *dst, const void *src, size_t size);
void memcpyD2H(void *dst, const void *src, size_t size);
void memcpyD2D(void *dst, const void *src, size_t size);
void memcpyH2H(void *dst, const void *src, size_t size);

} // namespace context

} // namespace infinicore
