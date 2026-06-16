local CUDNN_ROOT = os.getenv("CUDNN_ROOT") or os.getenv("CUDNN_HOME") or os.getenv("CUDNN_PATH")
if CUDNN_ROOT ~= nil then
    add_includedirs(CUDNN_ROOT .. "/include")
end

local CUTLASS_ROOT = os.getenv("CUTLASS_ROOT") or os.getenv("CUTLASS_HOME") or os.getenv("CUTLASS_PATH")
local TVM_ROOT = os.getenv("TVM_ROOT") or os.getenv("TVM_HOME") or os.getenv("TVM_PATH")

local FLASH_ATTN_ROOT = get_config("flash-attn")

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

-- Apply -gencode from `xmake f --cuda_arch=sm_80` (comma-separated values supported).
-- Returns true when explicit arch flags were added.
local function apply_cuda_arch_flags(add_fn)
    local arch_opt = get_config("cuda_arch")
    if not arch_opt or type(arch_opt) ~= "string" or arch_opt == "" then
        return false
    end
    for _, arch in ipairs(arch_opt:split(",")) do
        arch = arch:trim()
        if arch ~= "" then
            local compute = arch:gsub("sm_", "compute_")
            add_fn("-gencode=arch=" .. compute .. ",code=" .. arch)
        end
    end
    return true
end

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

        -- Auto-detect CUDA arch when no explicit --cuda_arch
        local arch_opt = get_config("cuda_arch")
        local script_path = path.join(
            os.projectdir(),
            "src/infiniop/ops/awq_marlin_gemm/nvidia/generate_kernels.py"
        )

        local header_path = path.join(
            os.projectdir(),
            "src/infiniop/ops/awq_marlin_gemm/nvidia/kernel_selector.h"
        )

        local cuda_arch_num = nil

        if arch_opt and type(arch_opt) == "string" then
            cuda_arch_num = arch_opt:match("sm_(%d+)")
        end

        local generate_arch =
            cuda_arch_num and (tonumber(cuda_arch_num) / 10) or 8.0

        if not os.isfile(header_path) then

            -- save current directory
            local oldir = os.curdir()

            -- switch cwd to script directory
            os.cd(path.directory(script_path))

            -- IMPORTANT:
            -- try = true prevents xmake abort
            local ok, errors = os.execv(
                "python",
                {
                    script_path,
                    tostring(generate_arch)
                },
                {
                    try = true
                }
            )

            -- restore cwd
            os.cd(oldir)

            if not ok then
                print("generate_kernels.py returned non-zero exit code")
                if errors then
                    print(errors)
                end
            end

            
        end

        if os.isfile(header_path) then
            print("AWQ Marlin kernels generated successfully!")
        else
            raise("Failed to generate AWQ Marlin kernels: header missing!")
        end

        -- CUDA arch: explicit --cuda_arch > nvidia-smi auto-detect > native
        if not apply_cuda_arch_flags(function(flag) target:add("cuflags", flag) end) then
            local ok, sm_str = os.iorunv("nvidia-smi", {"--query-gpu=compute_cap", "--format=csv,noheader,nounits"})
            if ok and sm_str then
                local major, minor = sm_str:match("(%d+)%.(%d+)")
                if major then
                    local sm = tonumber(major) * 10 + tonumber(minor)
                    local archs = {}
                    if sm >= 75 then table.insert(archs, "sm_75") end
                    if sm >= 80 then table.insert(archs, "sm_80") end
                    if sm >= 86 then table.insert(archs, "sm_86") end
                    if sm >= 89 then table.insert(archs, "sm_89") end
                    -- H100 (sm_90a): use sm_90a for cutlass 3.x
                    if sm == 90 then
                        target:add("cuflags", "-gencode=arch=compute_90a,code=sm_90a")
                    elseif sm > 90 then
                        table.insert(archs, "sm_90")
                    end
                    if #archs == 0 then
                        target:add("cugencodes", "native")
                    end
                    for _, arch in ipairs(archs) do
                        local compute = arch:gsub("sm_", "compute_")
                        target:add("cuflags", "-gencode=arch=" .. compute .. ",code=" .. arch)
                    end
                else
                    target:add("cugencodes", "native")
                end
            else
                target:add("cugencodes", "native")
            end
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
        add_cuflags("-Xcompiler=-fPIC", {force = true})
        add_cuflags("--extended-lambda")
        add_culdflags("-Xcompiler=-fPIC", {force = true})
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")
        add_cflags("-fPIC")
        add_cuflags("--expt-relaxed-constexpr")
        if CUDNN_ROOT ~= nil then
            add_linkdirs(CUDNN_ROOT .. "/lib")
        end
    end

    add_cuflags("-Xcompiler=-Wno-error=deprecated-declarations", "-Xcompiler=-Wno-error=unused-function")

    -- Cutlass: enable I8 Gemm when CUTLASS_ROOT is set
    if CUTLASS_ROOT ~= nil then
        add_defines("ENABLE_CUTLASS_API")
        add_includedirs(CUTLASS_ROOT, CUTLASS_ROOT .. "/include", CUTLASS_ROOT .. "/tools/util/include")
    end

    local arch_opt = get_config("cuda_arch")
    if TVM_ROOT ~= nil then
        add_defines("ENABLE_TVM_API")
        add_includedirs(TVM_ROOT, TVM_ROOT .. "/include", TVM_ROOT .. "/3rdparty/dlpack/include/")
        function parse_sgl_cuda_arch(arch)
    
            local num = arch:match("sm_(%d+)")
            if not num then
                return nil
            end

            return tonumber(num) * 10
        end
        if arch_opt then
            local sgl_arch = parse_sgl_cuda_arch(arch_opt)
            if sgl_arch then
                add_defines("SGL_CUDA_ARCH=" .. sgl_arch)
            else
                print("Invalid cuda_arch:", arch_opt)
            end
        else
            error("tvm complie marlin needs cuda_arch")
        end
    end
    
    if arch_opt and type(arch_opt) == "string" then
        for _, arch in ipairs(arch_opt:split(",")) do
            arch = arch:trim()
            local compute = arch:gsub("sm_", "compute_")
            add_cuflags("-gencode=arch=" .. compute .. ",code=" .. arch)
        end
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
        add_cuflags("-Xcompiler=-fPIC", {force = true})
        add_culdflags("-Xcompiler=-fPIC", {force = true})
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
            add_cuflags("-Xcompiler=-fPIC", {force = true})
            add_culdflags("-Xcompiler=-fPIC", {force = true})
            add_cxflags("-fPIC", {force = true})
            add_cxxflags("-fPIC", {force = true})

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

    on_load(function (target)
        if not apply_cuda_arch_flags(function(flag) target:add("cuflags", flag) end) then
            target:add("cugencodes", "native")
        end
    end)

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
            if CUTLASS_ROOT ~= nil then
                target:add("includedirs", CUTLASS_ROOT .. "/include", {public = false})
            end
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
