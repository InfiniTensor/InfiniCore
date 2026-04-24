
local MACA_ROOT = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or os.getenv("MACA_ROOT")
local FLASH_ATTN_ROOT = get_config("flash-attn")

-- MetaX flash-attn (pip `flash_attn_2_cuda`) may append an extra trailing argument
-- (`flash_attn_mars_ext_`) depending on the underlying HPCC/MetaX stack version.
do
    -- Intentionally empty: HPCC version parsing is deferred to `before_build`
    -- on `infinicore_cpp_api` where `os.iorunv` is available in this xmake sandbox.
end

-- Resolve MetaX flash-attn .so path (used only from this file: `before_link` sandbox cannot see globals from `xmake.lua`).
local FLASH_ATTN_METAX_CUDA_SO_CONTAINER_DEFAULT =
    "/opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so"

local function metax_flash_attn_cuda_so_path()
    -- Highest priority: override the exact `.so` file to link.
    local env_path = os.getenv("FLASH_ATTN_2_CUDA_SO")
    if env_path and env_path ~= "" then
        env_path = env_path:trim()
        if os.isfile(env_path) then
            return env_path
        end
        print(string.format("warning: metax+flash-attn: FLASH_ATTN_2_CUDA_SO is not a file: %s, fallback to container/default path", env_path))
    end

    -- Second priority: allow overriding the "expected" container path via env.
    local container_path = os.getenv("FLASH_ATTN_METAX_CUDA_SO_CONTAINER")
    if not container_path or container_path == "" then
        container_path = FLASH_ATTN_METAX_CUDA_SO_CONTAINER_DEFAULT
    end

    if not os.isfile(container_path) then
        print(
            string.format(
                "warning: metax+flash-attn: expected %s; install flash-attn in conda env, or export FLASH_ATTN_2_CUDA_SO.",
                container_path
            )
        )
    end
    return container_path
end

-- MetaX flash-attn link flags for pip `flash_attn_2_cuda`.
-- Version/ABI macros are set in `xmake.lua` for `infinicore_cpp_api` so they apply to all sources.
target("infinicore_cpp_api")
    if get_config("flash-attn") and get_config("flash-attn") ~= "" then
        before_link(function (target)
            local flash_so_metax = metax_flash_attn_cuda_so_path()
            local flash_dir_metax = path.directory(flash_so_metax)
            local flash_name_metax = path.filename(flash_so_metax)
            target:add(
                "shflags",
                "-Wl,--no-as-needed -L" .. flash_dir_metax .. " -l:" .. flash_name_metax .. " -Wl,-rpath," .. flash_dir_metax,
                {force = true}
            )
        end)
    end
target_end()

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
    add_cxflags("-lstdc++", "-fPIC", "-Wno-defaulted-function-deleted", "-Wno-strict-aliasing", {force = true})
    add_cxxflags("-lstdc++", "-fPIC", "-Wno-defaulted-function-deleted", "-Wno-strict-aliasing", {force = true})
    add_files("../src/infiniop/devices/metax/*.cc", "../src/infiniop/ops/*/metax/*.cc")
    add_files("../src/infiniop/ops/*/metax/*.maca", {rule = "maca"})

    if has_config("ninetoothed") then
        add_includedirs(MACA_ROOT .. "/include/hcr")
        add_includedirs(MACA_ROOT .. "/include/mcr")
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

target("flash-attn-metax")
    set_kind("phony")
    set_default(false)

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            local TORCH_DIR = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local PYTHON_INCLUDE = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_paths()['include'])"}):trim()
            local PYTHON_LIB_DIR = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"}):trim()

            -- Validate build/runtime env in container and keep these paths available for downstream linking.
            target:add("includedirs", TORCH_DIR .. "/include", TORCH_DIR .. "/include/torch/csrc/api/include", PYTHON_INCLUDE, {public = false})
            target:add("linkdirs", TORCH_DIR .. "/lib", PYTHON_LIB_DIR, {public = false})
        end)
    else
        before_build(function (target)
            print("Flash Attention not available, skipping flash-attn-metax integration")
        end)
    end
target_end()

target("infinirt-metax")
    set_kind("static")
    set_languages("cxx17")
    on_install(function (target) end)
    add_deps("infini-utils")
    set_warnings("all", "error")
    add_cxflags("-lstdc++ -fPIC")
    add_cxxflags("-lstdc++ -fPIC")
    add_files("../src/infinirt/metax/*.cc")
target_end()

target("infiniccl-metax")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")
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
