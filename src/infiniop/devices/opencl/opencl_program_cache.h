#ifndef __INFINIOP_OPENCL_PROGRAM_CACHE_H__
#define __INFINIOP_OPENCL_PROGRAM_CACHE_H__

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>

namespace device::opencl {

class ProgramCache {
public:
    ProgramCache() = default;
    ~ProgramCache() = default;

    std::shared_ptr<void> getOrBuildWithSource(
        const std::string &op_name,
        const std::string &source,
        const std::string &build_opts,
        cl_context context,
        cl_device_id device);

    ProgramCache(const ProgramCache &) = delete;
    ProgramCache &operator=(const ProgramCache &) = delete;

private:
    struct Entry {
        std::shared_ptr<void> program;
        bool building = false;
        bool failed = false;
        std::condition_variable cv;
    };
    mutable std::mutex mtx_;
    std::unordered_map<std::string, std::unique_ptr<Entry>> map_;
};

} // namespace device::opencl

#endif // __INFINIOP_OPENCL_PROGRAM_CACHE_H__
