add_defines("ENABLE_ASCEND_API")
local ASCEND_HOME = os.getenv("ASCEND_HOME") or os.getenv("ASCEND_TOOLKIT_HOME")
local SOC_VERSION = os.getenv("SOC_VERSION")

-- Add include dirs
add_includedirs(ASCEND_HOME .. "/include")
add_includedirs(ASCEND_HOME .. "/include/aclnn")
add_linkdirs(ASCEND_HOME .. "/lib64")
add_links("libascendcl.so")
add_links("libnnopbase.so")
add_links("libopapi.so")
add_links("libruntime.so")
add_linkdirs(ASCEND_HOME .. "/../../driver/lib64/driver")
add_links("libascend_hal.so")
local builddir = string.format(
        "%s/build/%s/%s/%s",
        os.projectdir(),
        get_config("plat"),
        get_config("arch"),
        get_config("mode")
    )

local function _newer_than(filepath, timestamp)
    if os.isfile(filepath) then
        local mtime = os.mtime(filepath)
        return mtime and mtime > timestamp
    end
    return false
end

local function _ascend_kernel_build_reason(ascend_build_dir, archive)
    if not os.isfile(archive) then
        return "missing libascend_kernels.a"
    end

    local archive_mtime = os.mtime(archive)
    local explicit_inputs = {
        path.join(ascend_build_dir, "CMakeLists.txt"),
        path.join(ascend_build_dir, "Makefile")
    }
    for _, filepath in ipairs(explicit_inputs) do
        if _newer_than(filepath, archive_mtime) then
            return filepath
        end
    end

    local input_patterns = {
        path.join(os.projectdir(), "src/infiniop/devices/ascend/*.h"),
        path.join(os.projectdir(), "src/infiniop/devices/ascend/*.hpp"),
        path.join(os.projectdir(), "src/infiniop/devices/ascend/*.cmake"),
        path.join(os.projectdir(), "src/infiniop/ops/*/ascend/*_kernel.cpp"),
        path.join(os.projectdir(), "src/infiniop/ops/*/ascend/*_kernel.h"),
        path.join(os.projectdir(), "src/infiniop/ops/*/ascend/*_kernel.hpp")
    }
    for _, pattern in ipairs(input_patterns) do
        for _, filepath in ipairs(os.files(pattern)) do
            if _newer_than(filepath, archive_mtime) then
                return filepath
            end
        end
    end

    return nil
end

rule("ascend-kernels")
    before_link(function ()
        local ascend_build_dir = path.join(os.projectdir(), "src/infiniop/devices/ascend")
        local ascend_archive = path.join(ascend_build_dir, "build/lib/libascend_kernels.a")
        local build_reason = _ascend_kernel_build_reason(ascend_build_dir, ascend_archive)
        os.cd(ascend_build_dir)
        if build_reason then
            print("building ascend kernels: " .. build_reason)
            local cmake_files_dir = path.join(
                ascend_build_dir,
                "build/ascend_kernels_preprocess-prefix/src/ascend_kernels_preprocess-build/CMakeFiles"
            )
            os.exec("rm -rf " .. path.join(cmake_files_dir, "aic_obj.dir") .. " " .. path.join(cmake_files_dir, "aiv_obj.dir"))
            os.exec("make build")
        else
            print("ascend kernels are up to date")
        end
        os.cp(ascend_archive, builddir.."/")
        os.cd(os.projectdir())

    end)
    after_clean(function ()
        local ascend_build_dir = path.join(os.projectdir(), "src/infiniop/devices/ascend")
        os.cd(ascend_build_dir)
        os.exec("make clean")
        os.cd(os.projectdir())
        os.exec("rm -f " .. builddir.. "/libascend_kernels.a")

    end)
rule_end()

target("infiniop-ascend")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    add_cxflags("-lstdc++ -fPIC")
    add_cxxflags("-lstdc++ -fPIC")
    set_warnings("all", "error")

    set_languages("cxx17")
    add_files("$(projectdir)/src/infiniop/devices/ascend/*.cc", "$(projectdir)/src/infiniop/ops/*/ascend/*.cc")

    -- Add operator
    add_rules("ascend-kernels")
    add_links(builddir.."/libascend_kernels.a")
target_end()

target("infinirt-ascend")
    set_kind("static")
    set_languages("cxx17")
    on_install(function (target) end)
    add_deps("infini-utils")
    -- Add files
    add_files("$(projectdir)/src/infinirt/ascend/*.cc")
    add_cxflags("-lstdc++ -Wall -Werror -fPIC")
    add_cxxflags("-lstdc++ -Wall -Werror -fPIC")
target_end()

target("infiniccl-ascend")
    set_kind("static")
    add_deps("infinirt")
    add_deps("infini-utils")
    set_warnings("all", "error")
    set_languages("cxx17")
    on_install(function (target) end)
    if has_config("ccl") then
        add_includedirs(ASCEND_HOME .. "/include/hccl")
        add_links("libhccl.so")
        add_files("../src/infiniccl/ascend/*.cc")
        add_cxflags("-lstdc++ -fPIC")
        add_cxxflags("-lstdc++ -fPIC")
    end
target_end()
