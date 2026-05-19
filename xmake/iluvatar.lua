local iluvatar_arch = get_config("iluvatar_arch") or "ivcore20"
local FLASH_ATTN_ROOT = get_config("flash-attn")

toolchain("iluvatar.toolchain")
    set_toolset("cc"  , "clang"  )
    set_toolset("cxx" , "clang++")
    set_toolset("cu"  , "clang++")
    set_toolset("culd", "clang++")
    set_toolset("cu-ccbin", "$(env CXX)", "$(env CC)")
toolchain_end()

rule("iluvatar.env")
    add_deps("cuda.env", {order = true})
    after_load(function (target)
        local old = target:get("syslinks")
        local new = {}

        for _, link in ipairs(old) do
            if link ~= "cudadevrt" then
                table.insert(new, link)
            end
        end

        if #old > #new then
            target:set("syslinks", new)
            local log = "cudadevrt removed, syslinks = { "
            for _, link in ipairs(new) do
                log = log .. link .. ", "
            end
            log = log:sub(0, -3) .. " }"
            print(log)
        end
    end)
rule_end()

target("infiniop-iluvatar")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("iluvatar.toolchain")
    add_rules("iluvatar.env")
    set_values("cuda.rdc", false)

    add_links("cudart", "cublas", "cudnn")

    set_warnings("all", "error")
    add_cuflags("-Wno-error=unused-private-field", "-Wno-error=unused-variable", "-Wno-unused-variable")
    add_cuflags("-fPIC", "-x", "ivcore", "-std=c++17", {force = true})
    add_cuflags("--cuda-gpu-arch=" .. iluvatar_arch, {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC", "-Wno-error=unused-variable", "-Wno-unused-variable")
    add_cxxflags("-fPIC", "-Wno-error=unused-variable", "-Wno-unused-variable")

    -- set_languages("cxx17") 天数似乎不能用这个配置
    add_files("../src/infiniop/devices/nvidia/*.cu", "../src/infiniop/ops/*/nvidia/*.cu")
    -- skip gaussian_nll_loss and hinge_embedding_loss and adapt them later
    remove_files("../src/infiniop/ops/gaussian_nll_loss/nvidia/*.cu")
    remove_files("../src/infiniop/ops/hinge_embedding_loss/nvidia/*.cu")

    add_files("../src/infiniop/ops/*/iluvatar/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", "../build/ninetoothed/*.cpp", {cxxflags = {"-Wno-return-type"}})
    end
target_end()

target("infinirt-iluvatar")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("iluvatar.toolchain")
    add_rules("iluvatar.env")
    set_values("cuda.rdc", false)

    add_links("cudart")

    set_warnings("all", "error")
    add_cuflags("-fPIC", "-x", "ivcore", "-std=c++17", {force = true})
    add_cuflags("--cuda-gpu-arch=" .. iluvatar_arch, {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")
    add_cxxflags("-fPIC")

    -- set_languages("cxx17") 天数似乎不能用这个配置
    add_files("../src/infinirt/cuda/*.cu")
target_end()

target("infiniccl-iluvatar")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)

    if has_config("ccl") then
        set_toolchains("iluvatar.toolchain")
        add_rules("iluvatar.env")
        set_values("cuda.rdc", false)

        add_links("cudart")

        set_warnings("all", "error")
        add_cuflags("-fPIC", "-x", "ivcore", "-std=c++17", {force = true})
        add_cuflags("--cuda-gpu-arch=" .. iluvatar_arch, {force = true})
        add_culdflags("-fPIC")
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")

        local nccl_root = os.getenv("NCCL_ROOT")
        if nccl_root then
            add_includedirs(nccl_root .. "/include")
            add_links(nccl_root .. "/lib/libnccl.so")
        else
            add_links("nccl") -- Fall back to default nccl linking
        end

        -- set_languages("cxx17") 天数似乎不能用这个配置
        add_files("../src/infiniccl/cuda/*.cu")
    end
target_end()

local function iluvatar_flash_attn_cuda_so_path()
    local env_path = os.getenv("FLASH_ATTN_2_CUDA_SO")
    if env_path and env_path ~= "" then
        env_path = env_path:trim()
        if os.isfile(env_path) then
            return env_path
        end
        print(string.format(
            "warning: iluvatar+flash-attn: FLASH_ATTN_2_CUDA_SO is not a file: %s, fallback to default path",
            env_path
        ))
    end

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        local files = os.files(path.join(FLASH_ATTN_ROOT, "flash_attn_2_cuda*.so"))
        if files and #files > 0 then
            return files[1]
        end
    end

    local container_path = os.getenv("FLASH_ATTN_ILUVATAR_CUDA_SO_CONTAINER")
    if container_path and container_path ~= "" and os.isfile(container_path) then
        return container_path:trim()
    end

    raise("iluvatar+flash-attn: cannot locate flash_attn_2_cuda; install it in current Python env or export FLASH_ATTN_2_CUDA_SO")
end

target("flash-attn-iluvatar")
    set_kind("phony")
    set_default(false)

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            local TORCH_DIR = os.iorunv("python", {
                "-c", "import torch, os; print(os.path.dirname(torch.__file__))"
            }):trim()
            local PYTHON_INCLUDE = os.iorunv("python", {
                "-c", "import sysconfig; print(sysconfig.get_paths()['include'])"
            }):trim()
            local PYTHON_LIB_DIR = os.iorunv("python", {
                "-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
            }):trim()

            target:add("includedirs",
                TORCH_DIR .. "/include",
                TORCH_DIR .. "/include/torch/csrc/api/include",
                PYTHON_INCLUDE,
                {public = false}
            )
            target:add("linkdirs", TORCH_DIR .. "/lib", PYTHON_LIB_DIR, {public = false})
        end)
    else
        before_build(function (target)
            print("Flash Attention not available, skipping flash-attn-iluvatar integration")
        end)
    end
target_end()
