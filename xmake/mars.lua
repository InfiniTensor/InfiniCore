local HPCC_ROOT = os.getenv("HPCC_PATH") or os.getenv("HPCC_HOME") or "/opt/hpcc"
local FLASH_ATTN_ROOT = get_config("flash-attn")

local FLASH_ATTN_MARS_CUDA_SO_CONTAINER_DEFAULT =
    "/opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-aarch64-linux-gnu.so"

local function mars_flash_attn_cuda_so_path()
    local env_path = os.getenv("FLASH_ATTN_2_CUDA_SO")
    if env_path and env_path ~= "" then
        env_path = env_path:trim()
        if os.isfile(env_path) then
            return env_path
        end
        print(string.format("warning: mars+flash-attn: FLASH_ATTN_2_CUDA_SO is not a file: %s", env_path))
    end

    local container_path = os.getenv("FLASH_ATTN_MARS_CUDA_SO_CONTAINER")
    if not container_path or container_path == "" then
        container_path = FLASH_ATTN_MARS_CUDA_SO_CONTAINER_DEFAULT
    end
    return container_path
end

target("infinicore_cpp_api")
    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_link(function (target)
            local flash_so = mars_flash_attn_cuda_so_path()
            local flash_dir = path.directory(flash_so)
            local flash_name = path.filename(flash_so)
            target:add(
                "shflags",
                "-Wl,--no-as-needed -L" .. flash_dir .. " -l:" .. flash_name .. " -Wl,-rpath," .. flash_dir,
                {force = true}
            )
        end)
    end
target_end()

add_includedirs(HPCC_ROOT .. "/include")
add_linkdirs(HPCC_ROOT .. "/lib")
add_links("hcdnn", "hcblas", "hcruntime")

rule("hpcc")
    set_extensions(".maca")

    on_load(function (target)
        target:add("includedirs", "include")
    end)

    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        local htcc = path.join(HPCC_ROOT, "htgpu_llvm/bin/htcc")
        local args = {
            "-x", "hpcc", "-c", sourcefile, "-o", objectfile,
            "-I" .. HPCC_ROOT .. "/include", "-O3", "-fPIC", "-Werror", "-std=c++17"
        }

        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end
        for _, define in ipairs(target:get("defines")) do
            table.insert(args, "-D" .. define)
        end

        table.insert(target:objectfiles(), objectfile)
        batchcmds:mkdir(path.directory(objectfile))
        batchcmds:show_progress(opt.progress, "${color.build.object}compiling.hpcc %s", sourcefile)
        batchcmds:vrunv(htcc, args)
        batchcmds:add_depfiles(sourcefile)
        batchcmds:set_depmtime(os.mtime(objectfile))
        batchcmds:set_depcache(target:dependfile(objectfile))
    end)
rule_end()

target("infiniop-mars")
    set_kind("static")
    on_install(function (target) end)
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cxflags("-lstdc++", "-fPIC", "-Wno-defaulted-function-deleted", "-Wno-strict-aliasing", {force = true})
    add_cxxflags("-lstdc++", "-fPIC", "-Wno-defaulted-function-deleted", "-Wno-strict-aliasing", {force = true})
    add_files("../src/infiniop/devices/metax/*.cc")
    add_files("../src/infiniop/ops/*/metax/*.cc")
    add_files("../src/infiniop/ops/*/metax/*.maca", {rule = "hpcc"})

    if has_config("ninetoothed") then
        add_includedirs(HPCC_ROOT .. "/include/hcr")
        add_files("../build/ninetoothed/*.c", "../build/ninetoothed/*.cpp", {
            cxflags = {
                "-include stdlib.h",
                "-Wno-return-type",
                "-Wno-implicit-function-declaration",
                "-Wno-builtin-declaration-mismatch"
            }
        })
    end
target_end()

target("flash-attn-mars")
    set_kind("phony")
    set_default(false)

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            local torch_dir = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local python_include = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_paths()['include'])"}):trim()
            local python_lib_dir = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"}):trim()
            target:add("includedirs", torch_dir .. "/include", torch_dir .. "/include/torch/csrc/api/include", python_include, {public = false})
            target:add("linkdirs", torch_dir .. "/lib", python_lib_dir, {public = false})
        end)
    end
target_end()

target("infiniccl-mars")
    set_kind("static")
    on_install(function (target) end)
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")
    end
    if has_config("ccl") then
        add_links("libhccl.so")
        add_files("../src/infiniccl/mars/*.cc")
    end
    set_languages("cxx17")
target_end()
