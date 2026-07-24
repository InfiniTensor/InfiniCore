add_defines("ENABLE_ASCEND_API")
local ASCEND_HOME = os.getenv("ASCEND_HOME") or os.getenv("ASCEND_TOOLKIT_HOME") or os.getenv("ASCEND_HOME_PATH")
local SOC_VERSION = os.getenv("SOC_VERSION")

-- Add include dirs
for _, include_dir in ipairs({
    path.join(ASCEND_HOME, "include"),
    path.join(ASCEND_HOME, "include/aclnn"),
    path.join(ASCEND_HOME, "aarch64-linux/include"),
    path.join(ASCEND_HOME, "aarch64-linux/include/aclnn"),
}) do
    if os.isdir(include_dir) then
        add_includedirs(include_dir)
    end
end
for _, lib_dir in ipairs({
    path.join(ASCEND_HOME, "lib64"),
    path.join(ASCEND_HOME, "aarch64-linux/lib64"),
}) do
    if os.isdir(lib_dir) then
        add_linkdirs(lib_dir)
    end
end
add_links("libascendcl.so")
add_links("libnnopbase.so")
add_links("libopapi.so")
add_links("libruntime.so")
for _, driver_dir in ipairs({
    path.join(ASCEND_HOME, "../driver/lib64/driver"),
    path.join(ASCEND_HOME, "../../driver/lib64/driver"),
}) do
    if os.isdir(driver_dir) then
        add_linkdirs(driver_dir)
    end
end
add_links("libascend_hal.so")
local builddir = string.format(
        "%s/build/%s/%s/%s",
        os.projectdir(),
        get_config("plat"),
        get_config("arch"),
        get_config("mode")
    )
rule("ascend-kernels")
    before_link(function ()
        local ascend_build_dir = path.join(os.projectdir(), "src/infiniop/devices/ascend")
        os.cd(ascend_build_dir)
        os.rm("build")
        os.exec("make build")
        os.cp("$(projectdir)/src/infiniop/devices/ascend/build/lib/libascend_kernels.a", builddir.."/")
        os.cd(os.projectdir())

    end)
    after_clean(function ()
        local ascend_build_dir = path.join(os.projectdir(), "src/infiniop/devices/ascend")
        os.cd(ascend_build_dir)
        os.exec("make clean")
        os.cd(os.projectdir())
        os.rm(builddir.. "/libascend_kernels.a")

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
    set_default(false)
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
