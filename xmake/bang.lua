
local NEUWARE_HOME = os.getenv("NEUWARE_HOME") or "/usr/local/neuware"
local FLASH_ATTN_ROOT = get_config("flash-attn")
add_includedirs(path.join(NEUWARE_HOME, "include"), {public = true})
add_linkdirs(path.join(NEUWARE_HOME, "lib64"))
add_linkdirs(path.join(NEUWARE_HOME, "lib"))
add_links("libcnrt.so")
add_links("libcnnl.so")
add_links("libcnnl_extra.so")
add_links("libcnpapi.so")

local FLASH_ATTN_CAMBRICON_BANG_SO_CONTAINER_DEFAULT =
    "/torch/venv3/pytorch/lib/python3.10/site-packages/flash_attn_2_bang.cpython-310-x86_64-linux-gnu.so"

local function cambricon_flash_attn_bang_so_path()
    local env_path = os.getenv("FLASH_ATTN_2_BANG_SO")
    if env_path and env_path ~= "" then
        env_path = env_path:trim()
        if os.isfile(env_path) then
            return env_path
        end
        print(string.format("warning: cambricon+flash-attn: FLASH_ATTN_2_BANG_SO is not a file: %s, fallback to python/container/default path", env_path))
    end

    local container_path = os.getenv("FLASH_ATTN_CAMBRICON_BANG_SO_CONTAINER")
    if container_path and container_path ~= "" then
        container_path = container_path:trim()
        if os.isfile(container_path) then
            return container_path
        end
        print(string.format("warning: cambricon+flash-attn: FLASH_ATTN_CAMBRICON_BANG_SO_CONTAINER is not a file: %s, fallback to python/default path", container_path))
    end

    if not os.isfile(FLASH_ATTN_CAMBRICON_BANG_SO_CONTAINER_DEFAULT) then
        print(
            string.format(
                "warning: cambricon+flash-attn: expected %s; install flash-attn in the container, or export FLASH_ATTN_2_BANG_SO.",
                FLASH_ATTN_CAMBRICON_BANG_SO_CONTAINER_DEFAULT
            )
        )
    end
    return FLASH_ATTN_CAMBRICON_BANG_SO_CONTAINER_DEFAULT
end

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
    add_cxxflags("-lstdc++ -fPIC")
    set_warnings("all", "error")

    set_languages("cxx17")
    add_files(src_dir.."/devices/bang/*.cc", src_dir.."/ops/*/bang/*.cc")
    local mlu_files = os.files(src_dir .. "/ops/*/bang/*.mlu")
    if #mlu_files > 0 then
        add_files(mlu_files, {rule = "mlu"})
    end
target_end()

target("flash-attn-cambricon")
    set_kind("phony")
    set_default(false)

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            local TORCH_DIR = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local TORCH_MLU_DIR = os.iorunv("python", {"-c", "import torch_mlu, os; print(os.path.dirname(torch_mlu.__file__))"}):trim()
            local PYTHON_INCLUDE = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_paths()['include'])"}):trim()
            local PYTHON_LIB_DIR = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"}):trim()

            target:add(
                "includedirs",
                TORCH_DIR .. "/include",
                TORCH_DIR .. "/include/torch/csrc/api/include",
                TORCH_MLU_DIR .. "/csrc/include",
                TORCH_MLU_DIR .. "/csrc/include/api/include",
                PYTHON_INCLUDE,
                {public = false}
            )
            target:add("linkdirs", TORCH_DIR .. "/lib", TORCH_MLU_DIR .. "/csrc/lib", PYTHON_LIB_DIR, {public = false})
        end)
    else
        before_build(function (target)
            print("Flash Attention not available, skipping flash-attn-cambricon integration")
        end)
    end
target_end()

if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
    target("infinicore_cpp_api")
        before_link(function (target)
            local flash_so_cambricon = cambricon_flash_attn_bang_so_path()
            local flash_dir_cambricon = path.directory(flash_so_cambricon)
            local flash_name_cambricon = path.filename(flash_so_cambricon)
            target:add(
                "shflags",
                "-Wl,--no-as-needed -L" .. flash_dir_cambricon .. " -l:" .. flash_name_cambricon .. " -Wl,-rpath," .. flash_dir_cambricon,
                {force = true}
            )
        end)
    target_end()
end

target("infinirt-cambricon")
    set_kind("static")
    add_deps("infini-utils")
    set_languages("cxx17")
    on_install(function (target) end)
    -- Add include dirs
    add_files("../src/infinirt/bang/*.cc")
    add_cxflags("-lstdc++ -Wall -Werror -fPIC")
    add_cxxflags("-lstdc++ -Wall -Werror -fPIC")
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
            add_cxxflags("-fPIC")
            add_ldflags("-fPIC")
        else
            print("[Warning] CNCL is currently only supported on Linux")
        end
    end
target_end()
