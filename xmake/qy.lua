local CUDNN_ROOT = os.getenv("CUDNN_ROOT") or os.getenv("CUDNN_HOME") or os.getenv("CUDNN_PATH")
if CUDNN_ROOT ~= nil then
    add_includedirs(CUDNN_ROOT .. "/include")
end

local FLASH_ATTN_ROOT = get_config("flash-attn")

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

function _qy_flash_attn_cuda_so_path()
    -- Highest priority: override the exact `.so` file to link.
    local env_path = os.getenv("FLASH_ATTN_2_CUDA_SO")
    if env_path and env_path ~= "" then
        env_path = env_path:trim()
        if os.isfile(env_path) then
            return env_path
        end
        print(string.format("warning: qy+flash-attn: FLASH_ATTN_2_CUDA_SO is not a file: %s, fallback to container/default path", env_path))
    end

    -- Second priority: allow overriding the "expected" container path via env.
    local container_path = os.getenv("FLASH_ATTN_QY_CUDA_SO_CONTAINER")
    if not container_path or container_path == "" then
        raise("Error: Flash Attention SO path not specified!\n")
end

    if not os.isfile(container_path) then
        print(
            string.format(
                "warning: qy+flash-attn: expected %s; install flash-attn in conda env, or export FLASH_ATTN_2_CUDA_SO.",
                container_path
            )
        )
    end
    return container_path
end

add_includedirs("/usr/local/denglin/sdk/include", "../include")
add_linkdirs("/usr/local/denglin/sdk/lib")
add_links("curt", "cublas", "cudnn")
set_languages("cxx17")
add_cxxflags("-std=c++17")  -- 显式设置 C++17
add_cuflags("--std=c++17",{force = true})  -- 确保 CUDA 编译器也使用 C++17
rule("ignore.o")
    set_extensions(".o")  -- 防止 xmake 默认处理
    on_build_files(function () end)

rule("qy.cuda")
    set_extensions(".cu")

    -- 缓存所有 .o 文件路径
    local qy_objfiles = {}

    on_load(function (target)
        target:add("includedirs", "/usr/local/denglin/sdk/include")
    end)

    after_load(function (target)
        -- 过滤 cudadevrt/cudart_static
        local links = target:get("syslinks") or {}
        local filtered = {}
        for _, link in ipairs(links) do
            if link ~= "cudadevrt" and link ~= "cudart_static" then
                table.insert(filtered, link)
            end
        end
        target:set("syslinks", filtered)
    end)

    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        import("core.project.project")
        import("core.project.config")
        import("core.base.option")

        local dlcc = "/usr/local/denglin/sdk/bin/dlcc"
        local sdk_path = "/usr/local/denglin/sdk"
        local arch = "dlgput64"

        
        local relpath = path.relative(sourcefile, os.projectdir())

        relpath = relpath:gsub("%.%.", "__")

        local objfile = path.join(
            config.buildir(),
            ".objs",
            target:name(),
            "rules",
            "qy.cuda",
            relpath .. ".o"
        )

        target:add("objectfiles", objfile)
        target:set("buildadd", true)
        local argv = {
            "-c", sourcefile,
            "-o", objfile,
            "--cuda-path=" .. sdk_path,
            "--cuda-gpu-arch=" .. arch,
            "-std=c++17", "-O2", "-fPIC"
        }

        for _, incdir in ipairs(target:get("includedirs") or {}) do
            table.insert(argv, "-I" .. incdir)
        end
        for _, def in ipairs(target:get("defines") or {}) do
            table.insert(argv, "-D" .. def)
        end

        batchcmds:mkdir(path.directory(objfile))
        batchcmds:show_progress(opt.progress, "${color.build.object}compiling.dlcu %s", relpath)
        batchcmds:vrunv(dlcc, argv)
    end)
target("infiniop-qy")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    add_rules("qy.cuda", {override = true})

    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cuflags("-Xcompiler=/W3", "-Xcompiler=/WX")
        add_cxxflags("/FS")
        if CUDNN_ROOT ~= nil then
            add_linkdirs(CUDNN_ROOT .. "\\lib\\x64")
        end
    else
        add_cuflags("-Xcompiler=-Wall", "-Xcompiler=-Werror")
        add_cuflags("-Xcompiler=-fPIC")
        add_cuflags("--extended-lambda")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")
        add_cuflags("--expt-relaxed-constexpr")
        if CUDNN_ROOT ~= nil then
            add_linkdirs(CUDNN_ROOT .. "/lib")
        end
    end

    add_cuflags("-Xcompiler=-Wno-error=deprecated-declarations")

    set_languages("cxx17")
    add_files("../src/infiniop/devices/nvidia/*.cu", "../src/infiniop/ops/*/nvidia/*.cu", "../src/infiniop/ops/*/*/nvidia/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", "../build/ninetoothed/*.cpp")
    end
target_end()

target("infinirt-qy")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)
    add_rules("qy.cuda", {override = true})
    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cxxflags("/FS")
    else
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")
    end

    set_languages("cxx17")
    add_files("../src/infinirt/cuda/*.cu")
target_end()

target("infiniccl-qy")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)
    if has_config("ccl") then
        add_rules("qy.cuda", {override = true})
        if not is_plat("windows") then
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
            add_cxflags("-fPIC")
            add_cxxflags("-fPIC")

            local nccl_root = os.getenv("NCCL_ROOT")
            if nccl_root then
                add_includedirs(nccl_root .. "/include")
                add_links(nccl_root .. "/lib/libnccl.so")
            else
                add_links("nccl") -- Fall back to default nccl linking
            end

            add_files("../src/infiniccl/cuda/*.cu")
        else
            print("[Warning] NCCL is not supported on Windows")
        end
    end
    set_languages("cxx17")

target_end()

target("flash-attn-qy")
    set_kind("phony")
    set_default(false)
    

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            target:add("includedirs", "/usr/local/denglin/sdk/include", {public = true})
            local TORCH_DIR = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local PYTHON_INCLUDE = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_paths()['include'])"}):trim()
            local PYTHON_LIB_DIR = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"}):trim()
            
            -- Validate build/runtime env in container and keep these paths available for downstream linking.
            target:add("includedirs", TORCH_DIR .. "/include", TORCH_DIR .. "/include/torch/csrc/api/include", PYTHON_INCLUDE, {public = false})
            target:add("linkdirs", TORCH_DIR .. "/lib", PYTHON_LIB_DIR, {public = false})
        end)
    else
        before_build(function (target)
            print("Flash Attention not available, skipping flash-attn-qy integration")
        end)
    end
target_end()
