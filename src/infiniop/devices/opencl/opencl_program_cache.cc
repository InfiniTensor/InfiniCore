#include "opencl_program_cache.h"
#include <iostream>
#include <sstream>

namespace device::opencl {

std::shared_ptr<void> ProgramCache::getOrBuildWithSource(
    const std::string &op_name,
    const std::string &source,
    const std::string &build_opts,
    cl_context context,
    cl_device_id device) {

    std::ostringstream oss;
    oss << build_opts << "#dev:" << (uintptr_t)device << "#ctx:" << (uintptr_t)context
        << "#op_name:" << op_name;
    std::string key = oss.str();

    std::unique_lock<std::mutex> lk(mtx_);
    auto &entry_ptr = map_[key];
    if (!entry_ptr) {
        entry_ptr.reset(new Entry());
    }
    Entry *entry = entry_ptr.get();

    if (entry->program) {
        return entry->program;
    }
    if (entry->failed) {
        return nullptr;
    }
    if (entry->building) {
        entry->cv.wait(lk, [&] { return !entry->building; });
        return entry->program;
    }

    entry->building = true;
    lk.unlock();

    cl_program raw_prog = nullptr;
    cl_int clerr = CL_SUCCESS;
    const char *src_ptr = source.c_str();
    size_t src_len = source.size();

    raw_prog = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &clerr);
    if (raw_prog == nullptr || clerr != CL_SUCCESS) {
        raw_prog = nullptr;
    } else {
        clerr = clBuildProgram(raw_prog, 1, &device, build_opts.c_str(), nullptr, nullptr);
        if (clerr != CL_SUCCESS) {
            // print build log
            size_t log_size = 0;
            clGetProgramBuildInfo(raw_prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            if (log_size > 0) {
                std::vector<char> log(log_size + 1);
                clGetProgramBuildInfo(raw_prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                log[log_size] = '\0';
                std::cerr << "[OpenCL] build log (" << op_name << "): " << log.data() << std::endl;
            } else {
                std::cerr << "[OpenCL] build failed for op " << op_name << ", clBuildProgram returned " << clerr << std::endl;
            }
            clReleaseProgram(raw_prog);
            raw_prog = nullptr;
        } else {
        }
    }

    lk.lock();
    if (raw_prog == nullptr) {
        entry->building = false;
        entry->failed = true;
        entry->cv.notify_all();
        return nullptr;
    } else {
        auto deleter = [](void *p) { if (p){ clReleaseProgram(reinterpret_cast<cl_program>(p));
} };
        std::shared_ptr<void> prog_shared(reinterpret_cast<void *>(raw_prog), deleter);
        entry->program = prog_shared;
        entry->building = false;
        entry->cv.notify_all();
        return entry->program;
    }
}

} // namespace device::opencl
