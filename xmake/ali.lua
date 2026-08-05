local FLASH_ATTN_ROOT = get_config("flash-attn")

local function ali_flash_attn_cuda_so_path()
    local env_path = os.getenv("FLASH_ATTN_2_CUDA_SO")
    if env_path and env_path ~= "" then
        env_path = env_path:trim()
        if os.isfile(env_path) then
            return env_path
        end
        print(string.format(
            "warning: ali+flash-attn: FLASH_ATTN_2_CUDA_SO is not a file: %s; falling back to --flash-attn",
            env_path))
    end

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        local candidates = os.files(path.join(FLASH_ATTN_ROOT, "flash_attn_2_cuda*.so"))
        if #candidates > 0 then
            return candidates[1]
        end
    end

    local detected = os.iorunv("python", {
        "-c",
        "import flash_attn_2_cuda; print(flash_attn_2_cuda.__file__)",
    }):trim()
    if detected ~= "" and os.isfile(detected) then
        return detected
    end

    raise("ali+flash-attn: flash_attn_2_cuda extension not found; pass its site-packages directory via --flash-attn or set FLASH_ATTN_2_CUDA_SO")
end

target("flash-attn-ali")
    set_kind("phony")
    set_default(false)
    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            local flash_so = ali_flash_attn_cuda_so_path()
            print("Ali PPU flash-attn extension: " .. flash_so)
        end)
    end
target_end()

target("infinicore_cpp_api")
    if has_config("aten") then
        before_link(function (target)
            local torch_dir = os.iorunv("python", {
                "-c",
                "import torch, os; print(os.path.dirname(torch.__file__))",
            }):trim()
            local torch_lib_dir = path.join(torch_dir, "lib")
            if not os.isdir(torch_lib_dir) then
                raise("ali+aten: torch library directory not found: " .. torch_lib_dir)
            end

            -- Ali's Torch libraries are loaded with local symbol visibility by
            -- Python. Link them explicitly so ATen calls from InfiniCore resolve
            -- even when no flash-attn extension is configured.
            target:add(
                "shflags",
                "-Wl,--no-as-needed",
                "-L" .. torch_lib_dir,
                "-ltorch_python",
                "-ltorch_cpu",
                "-ltorch_cuda",
                "-ltorch",
                "-lc10_cuda",
                "-lc10",
                "-Wl,--as-needed",
                "-Wl,-rpath," .. torch_lib_dir,
                {force = true}
            )

            if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
                local flash_so = ali_flash_attn_cuda_so_path()
                local flash_dir = path.directory(flash_so)
                local flash_name = path.filename(flash_so)
                target:add(
                    "shflags",
                    "-Wl,--no-as-needed",
                    "-L" .. flash_dir,
                    "-l:" .. flash_name,
                    "-Wl,--as-needed",
                    "-Wl,-rpath," .. flash_dir,
                    {force = true}
                )
            end
        end)
    end
target_end()

local CUDNN_ROOT = os.getenv("CUDNN_ROOT") or os.getenv("CUDNN_HOME") or os.getenv("CUDNN_PATH")
if CUDNN_ROOT ~= nil then
    add_includedirs(CUDNN_ROOT .. "/include")
end

local CUTLASS_ROOT = os.getenv("CUTLASS_ROOT") or os.getenv("CUTLASS_HOME") or os.getenv("CUTLASS_PATH")

if CUTLASS_ROOT ~= nil then
    add_includedirs(CUTLASS_ROOT)
end

target("infiniop-ali")
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
    add_files("../src/infiniop/devices/nvidia/*.cu", "../src/infiniop/ops/*/nvidia/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", "../build/ninetoothed/*.cpp")
    end
target_end()

target("infinirt-ali")
    set_kind("static")
    set_default(false)
    add_deps("infini-utils")
    on_install(function (target) end)

    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart")

    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cxxflags("/FS")
    else
        add_cuflags("-Xcompiler=-fPIC", "-Xcompiler=-shared")
        add_culdflags("-Xcompiler=-fPIC", "-Xcompiler=-shared")
        add_cxflags("-fPIC", "-shared")
        add_cxxflags("-fPIC", "-shared")
        add_shflags("-fPIC")
    end

    set_languages("cxx17")
    add_files("../src/infinirt/cuda/*.cu")
target_end()

target("infiniccl-ali")
    set_kind("static")
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
