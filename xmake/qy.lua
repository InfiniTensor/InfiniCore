local CUDNN_ROOT = os.getenv("CUDNN_ROOT") or os.getenv("CUDNN_HOME") or os.getenv("CUDNN_PATH")
if CUDNN_ROOT ~= nil then
    add_includedirs(CUDNN_ROOT .. "/include")
end

local CUTLASS_ROOT = os.getenv("CUTLASS_ROOT") or os.getenv("CUTLASS_HOME") or os.getenv("CUTLASS_PATH")

if CUTLASS_ROOT ~= nil then
    add_includedirs(CUTLASS_ROOT)
end

local FLASH_ATTN_ROOT = get_config("flash-attn")

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

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

        -- 去掉 ..，转成安全路径
        relpath = relpath:gsub("%.%.", "__")

        local objfile = path.join(
            config.buildir(),
            ".objs",
            target:name(),
            "rules",
            "qy.cuda",
            relpath .. ".o"
        )

        -- 🟢 强制注册 .o 文件给 target
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
    set_kind("shared")
    set_default(false)

    set_languages("cxx17")
    add_cxxflags("-std=c++17")
    add_cuflags("--std=c++17", {force = true})

    -- 🔥 DLCC 规则
    add_rules("qy.cuda", {override = true})

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= false and FLASH_ATTN_ROOT ~= "" then

        -- ⭐⭐⭐ 关键：用 on_load（不是 before_build）
        on_load(function (target)

            local TORCH_DIR = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local PYTHON_INCLUDE = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_paths()['include'])"}):trim()
            local PYTHON_LIB_DIR = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"}):trim()
            local LIB_PYTHON = os.iorunv("python", {"-c", "import glob,sysconfig,os;print(glob.glob(os.path.join(sysconfig.get_config_var('LIBDIR'),'libpython*.so'))[0])"}):trim()

            -- ✅ CUDA（最关键）
            target:add("includedirs", "/usr/local/denglin/sdk/include", {public = true})

            -- ✅ flash-attn
            target:add("includedirs", FLASH_ATTN_ROOT .. "/csrc")
            target:add("includedirs", FLASH_ATTN_ROOT .. "/csrc/flash_attn")
            target:add("includedirs", FLASH_ATTN_ROOT .. "/csrc/flash_attn/src")
            target:add("includedirs", FLASH_ATTN_ROOT .. "/csrc/common")

            -- ✅ torch
            target:add("includedirs", TORCH_DIR .. "/include")
            target:add("includedirs", TORCH_DIR .. "/include/torch/csrc/api/include")

            -- ⚠️ 很关键：ATen 有些头在这里
            target:add("includedirs", TORCH_DIR .. "/include/TH")
            target:add("includedirs", TORCH_DIR .. "/include/THC")

            -- ✅ python
            target:add("includedirs", PYTHON_INCLUDE)

            -- ✅ cutlass
            if CUTLASS_ROOT then
                target:add("includedirs", CUTLASS_ROOT .. "/include")
            end

            -- link dirs
            target:add("linkdirs", TORCH_DIR .. "/lib")
            target:add("linkdirs", PYTHON_LIB_DIR)
            target:add("linkdirs", "/usr/local/denglin/sdk/lib")

            -- links
            target:add("links",
                "curt",
                "cublas",
                "cudnn",
                "torch",
                "torch_cpu",
                "torch_cuda",
                "c10",
                "c10_cuda",
                "torch_python",
                LIB_PYTHON
            )
        end)

        -- ✅ C++ host
        add_files(FLASH_ATTN_ROOT .. "/csrc/flash_attn/flash_api.cpp")

        -- ✅ CUDA kernel
        add_files(FLASH_ATTN_ROOT .. "/csrc/flash_attn/src/*.cu")

        -- flags
        add_cxflags("-fPIC", {force = true})
        add_cuflags("-O2", "-fPIC", "--expt-relaxed-constexpr", "--use_fast_math", {force = true})

        add_ldflags("-Wl,--no-undefined", {force = true})

    else
        on_load(function ()
            print("Flash Attention not available, skipping flash-attn-qy build")
        end)
    end

    on_install(function (target) end)

target_end()
