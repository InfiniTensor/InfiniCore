
local MACA_ROOT = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or os.getenv("MACA_ROOT")
add_includedirs(MACA_ROOT .. "/include")
add_linkdirs(MACA_ROOT .. "/lib")
if has_config("use-mc") then
    add_links("mcdnn", "mcblas", "mcruntime")
else
    add_links("hcdnn", "hcblas", "hcruntime")
end

rule("maca")
    set_extensions(".maca")

    on_load(function (target)
        target:add("includedirs", "include")
    end)

    on_build_file(function (target, sourcefile)
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))
        local args
        local htcc
        if has_config("use-mc") then
            htcc = path.join(MACA_ROOT, "mxgpu_llvm/bin/mxcc")
            args = { "-x", "maca", "-c", sourcefile, "-o", objectfile, "-I" .. MACA_ROOT .. "/include", "-O3", "-fPIC", "-Werror", "-std=c++17"}
        else
            htcc = path.join(MACA_ROOT, "htgpu_llvm/bin/htcc")
            args = { "-x", "hpcc", "-c", sourcefile, "-o", objectfile, "-I" .. MACA_ROOT .. "/include", "-O3", "-fPIC", "-Werror", "-std=c++17"}
        end
        local includedirs = table.concat(target:get("includedirs"), " ")
        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end

        local defines = target:get("defines")
        for _, define in ipairs(defines) do
            table.insert(args, "-D" .. define)
        end

        os.execv(htcc, args)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("infiniop-metax")
    set_kind("static")
    on_install(function (target) end)
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cxflags("-lstdc++", "-fPIC", "-Wno-defaulted-function-deleted", "-Wno-strict-aliasing")
    add_files("../src/infiniop/devices/metax/*.cc", "../src/infiniop/ops/*/metax/*.cc")
    add_files("../src/infiniop/ops/*/metax/*.maca", {rule = "maca"})

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", {cxflags = {"-include stdlib.h", "-Wno-return-type"}})
    end
target_end()

target("infinirt-metax")
    set_kind("static")
    set_languages("cxx17")
    on_install(function (target) end)
    add_deps("infini-utils")
    set_warnings("all", "error")
    add_cxflags("-lstdc++ -fPIC")
    add_files("../src/infinirt/metax/*.cc")
target_end()

target("infiniccl-metax")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end
    if has_config("ccl") then
        if has_config("use-mc") then
            add_links("libmccl.so")
        else
            add_links("libhccl.so")
        end
        add_files("../src/infiniccl/metax/*.cc")
    end
    set_languages("cxx17")

target_end()
