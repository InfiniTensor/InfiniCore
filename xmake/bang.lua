
local NEUWARE_HOME = os.getenv("NEUWARE_HOME") or "/usr/local/neuware"
add_includedirs(path.join(NEUWARE_HOME, "include"), {public = true})
add_linkdirs(path.join(NEUWARE_HOME, "lib64"))
add_linkdirs(path.join(NEUWARE_HOME, "lib"))
add_links("libcnrt.so")
add_links("libcnnl.so")
add_links("libcnnl_extra.so")
add_links("libcnpapi.so")

rule("mlu")
    set_extensions(".mlu")

    on_load(function (target)
        target:add("includedirs", path.join(os.projectdir(), "include"))
    end)

    on_build_file(function (target, sourcefile)
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))

        local cc = "cncc"

        local includedirs = table.concat(target:get("includedirs"), " ")
        local args = {"-c", sourcefile, "-o", objectfile, "--bang-mlu-arch=mtp_592", "-O3", "-fPIC", "-Wall", "-Werror", "-std=c++17", "-pthread"}

        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end

        os.execv(cc, args)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

local src_dir = path.join(os.projectdir(), "src", "infiniop")

target("infiniop-cambricon")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    add_cxflags("-lstdc++ -fPIC")
    set_warnings("all", "error")

    set_languages("cxx17")
    add_files(src_dir.."/devices/bang/*.cc", src_dir.."/ops/*/bang/*.cc")
    local mlu_files = os.files(src_dir .. "/ops/*/bang/*.mlu")
    if #mlu_files > 0 then
        add_files(mlu_files, {rule = "mlu"})
    end
target_end()

target("infinirt-cambricon")
    set_kind("static")
    add_deps("infini-utils")
    set_languages("cxx17")
    on_install(function (target) end)
    -- Add include dirs
    add_files("../src/infinirt/bang/*.cc")
    add_cxflags("-lstdc++ -Wall -Werror -fPIC")
target_end()

target("infiniccl-cambricon")
    set_kind("static")
    add_deps("infinirt")
    add_deps("infini-utils")
    set_warnings("all", "error")
    set_languages("cxx17")
    on_install(function (target) end)
    
    if has_config("ccl") then
        if is_plat("linux") then
            add_includedirs(NEUWARE_HOME .. "/include")
            add_linkdirs(NEUWARE_HOME .. "/lib64")
            add_links("cncl", "cnrt")

            if has_package("libibverbs") then
                add_links("ibverbs")
                add_defines("CNCL_RDMA_ENABLED=1")
            end

            if is_arch("arm64") then
                add_defines("CNCL_ARM64_COMPAT_MODE=1")
            end

            add_rpathdirs(NEUWARE_HOME .. "/lib64")
            add_runenvs("LD_LIBRARY_PATH", NEUWARE_HOME .. "/lib64")

            add_files("../src/infiniccl/cambricon/*.cc")
            add_cxflags("-fPIC")
            add_ldflags("-fPIC")
        else
            print("[Warning] CNCL is currently only supported on Linux")
        end
    end
target_end()
