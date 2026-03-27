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

target("infiniop-nvidia")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart", "cublas")
    if has_config("cudnn") then
        add_links("cudnn")
    end

    on_load(function (target)
        import("lib.detect.find_tool")
        local nvcc = find_tool("nvcc")
        if nvcc ~= nil then
            if is_plat("windows") then
                nvcc_path = os.iorun("where nvcc"):match("(.-)\r?\n")
            else
                nvcc_path = nvcc.program
            end

            target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs")
            target:add("links", "cuda")
        end
    end)

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
        add_cflags("-fPIC")
        add_cuflags("--expt-relaxed-constexpr")
        if CUDNN_ROOT ~= nil then
            add_linkdirs(CUDNN_ROOT .. "/lib")
        end
    end

    add_cuflags("-Xcompiler=-Wno-error=deprecated-declarations", "-Xcompiler=-Wno-error=unused-function")

    local arch_opt = get_config("cuda_arch")
    if arch_opt and type(arch_opt) == "string" then
        for _, arch in ipairs(arch_opt:split(",")) do
            arch = arch:trim()
            local compute = arch:gsub("sm_", "compute_")
            add_cuflags("-gencode=arch=" .. compute .. ",code=" .. arch)
        end
    else
        add_cugencodes("native")
    end

    set_languages("cxx17")
    add_files("../src/infiniop/devices/nvidia/*.cu", "../src/infiniop/ops/*/nvidia/*.cu", "../src/infiniop/ops/*/*/nvidia/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", "../build/ninetoothed/*.cpp")
    end
target_end()

target("infinirt-nvidia")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart")

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

target("infiniccl-nvidia")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)
    if has_config("ccl") then
        set_policy("build.cuda.devlink", true)
        set_toolchains("cuda")
        add_links("cudart")

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

target("flash-attn-nvidia")
    set_kind("shared")
    set_default(false)
    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart")
    add_cugencodes("native")

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            local TORCH_DIR = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local PYTHON_INCLUDE = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_paths()['include'])"}):trim()
            local PYTHON_LIB_DIR = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"}):trim()
            local LIB_PYTHON = os.iorunv("python", {"-c", "import glob,sysconfig,os;print(glob.glob(os.path.join(sysconfig.get_config_var('LIBDIR'),'libpython*.so'))[0])"}):trim()
            
            -- Include dirs (needed for both device and host)
            target:add("includedirs", FLASH_ATTN_ROOT .. "/csrc/flash_attn/src", {public = false})
            target:add("includedirs", TORCH_DIR .. "/include/torch/csrc/api/include", {public = false})
            target:add("includedirs", TORCH_DIR .. "/include", {public = false})
            target:add("includedirs", PYTHON_INCLUDE, {public = false})
            target:add("includedirs", CUTLASS_ROOT .. "/include", {public = false})
            target:add("includedirs", FLASH_ATTN_ROOT .. "/csrc/flash_attn", {public = false})

            -- Link libraries
            target:add("linkdirs", TORCH_DIR .. "/lib", PYTHON_LIB_DIR)
            target:add("links", "torch", "torch_cuda", "torch_cpu", "c10", "c10_cuda", "torch_python", LIB_PYTHON)
        end)

        add_files(FLASH_ATTN_ROOT .. "/csrc/flash_attn/flash_api.cpp")
        add_files(FLASH_ATTN_ROOT .. "/csrc/flash_attn/src/*.cu")
        
        -- Link options
        add_ldflags("-Wl,--no-undefined", {force = true})
        
        -- Compile options
        add_cxflags("-fPIC", {force = true})
        add_cuflags("-Xcompiler=-fPIC")
        add_cuflags("--forward-unknown-to-host-compiler --expt-relaxed-constexpr --use_fast_math", {force = true})
        set_values("cuda.rdc", false)
    else
        -- If flash-attn is not available, just create an empty target
        before_build(function (target)
            print("Flash Attention not available, skipping flash-attn-nvidia build")
        end)
        on_build(function (target) end)
    end

    on_install(function (target) end)

target_end()
