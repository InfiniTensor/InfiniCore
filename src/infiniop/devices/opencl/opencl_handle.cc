#include "../../../infinirt/opencl/infinirt_opencl.h"
#include "opencl_common.h"

namespace device::opencl {
Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>(device_id)) {}

Handle::Handle(int device_id) : Handle(INFINI_DEVICE_OPENCL, device_id) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

Handle::Internal::Internal(int device_id) {
    infinirtInit();
    cl_device_id cl_device;
    infinirtOpenclDevice_t device;
    infinirtGetOpenclDevice(&device);
    cl_device = static_cast<cl_device_id>(device);

#if defined(INTEL)
    _warp_size = 32;
#elif defined(ADRENO)
    _warp_size = 128;
#else
    _warp_size = 32;
#endif

    size_t device_max_wg = 0;
    clGetDeviceInfo(cl_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_max_wg), &device_max_wg, nullptr);
    _max_threads_per_block = static_cast<int>(device_max_wg);

    size_t max_item_sizes[3];
    clGetDeviceInfo(cl_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_item_sizes), max_item_sizes, nullptr);
    _block_size[0] = max_item_sizes[0];
    _block_size[1] = max_item_sizes[1];
    _block_size[2] = max_item_sizes[2];
    program_cache_ = std::make_unique<ProgramCache>();
}

int Handle::Internal::warpSize() const { return _warp_size; }

int Handle::Internal::maxThreadsPerBlock() const { return _max_threads_per_block; }

int Handle::Internal::blockSizeX() const { return _block_size[0]; }

int Handle::Internal::blockSizeY() const { return _block_size[1]; }

int Handle::Internal::blockSizeZ() const { return _block_size[2]; }

ProgramCache *Handle::Internal::programCache() const { return program_cache_.get(); }

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(INFINI_DEVICE_OPENCL, device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::opencl
