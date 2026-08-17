local ILUVATAR_FLASH_ATTN_ROOT = get_config("flash-attn")

local iluvatar_flash_attn_enabled = has_config("use-vendor-ops")
    or (ILUVATAR_FLASH_ATTN_ROOT and ILUVATAR_FLASH_ATTN_ROOT ~= "")

local function iluvatar_attention_so_path(iorunv)
    if ILUVATAR_FLASH_ATTN_ROOT and ILUVATAR_FLASH_ATTN_ROOT ~= "" then
        local candidates = os.files(path.join(
            ILUVATAR_FLASH_ATTN_ROOT,
            "_C.cpython-*.so"
        ))
        if #candidates == 0 then
            candidates = os.files(path.join(ILUVATAR_FLASH_ATTN_ROOT, "_C*.so"))
        end
        if #candidates > 0 then
            return candidates[1]
        end

        raise("Iluvatar attention extension _C.cpython-*.so was not found under: "
            .. ILUVATAR_FLASH_ATTN_ROOT)
    end

    local detected = iorunv("python3", {
        "-c",
        "import glob, os, vllm_iluvatar; "
            .. "print(next(iter(glob.glob(os.path.join(os.path.dirname(vllm_iluvatar.__file__), '_C.cpython-*.so'))), ''))",
    }):trim()
    if detected ~= "" and os.isfile(detected) then
        return detected
    end

    raise("Iluvatar attention extension was not found; pass the vllm_iluvatar package directory via --flash-attn")
end

if iluvatar_flash_attn_enabled and not has_config("aten") then
    raise("Iluvatar Flash Attention requires --aten=true")
end

target("infinicore_cpp_api")
    if iluvatar_flash_attn_enabled then
        add_defines("ENABLE_ILUVATAR_FLASH_ATTN")
    end

    if has_config("use-vendor-ops") then
        add_defines("ENABLE_ILUVATAR_VENDOR_OPS")
    end

    if has_config("aten") then
        before_link(function (target)
            local torch_dir = os.iorunv("python3", {
                "-c",
                "import torch, os; print(os.path.dirname(torch.__file__))",
            }):trim()
            local torch_lib_dir = path.join(torch_dir, "lib")
            if not os.isdir(torch_lib_dir) then
                raise("Iluvatar ATen: torch library directory not found: " .. torch_lib_dir)
            end

            -- ATen-backed InfiniOps operators, Flash Attention, and vendor
            -- extensions use Torch symbols. Keep the complete Torch runtime
            -- discoverable even when users import infinicore before importing torch.
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

            if not iluvatar_flash_attn_enabled then
                return
            end

            local attention_so = iluvatar_attention_so_path(os.iorunv)
            print("Iluvatar attention extension: " .. attention_so)
            local attention_dir = path.directory(attention_so)
            local attention_name = path.filename(attention_so)
            target:add(
                "shflags",
                "-Wl,--no-as-needed",
                "-L" .. attention_dir,
                "-l:" .. attention_name,
                "-Wl,--as-needed",
                "-Wl,-rpath," .. attention_dir,
                {force = true}
            )
        end)
    end
target_end()

target("_infinicore")
    if has_config("use-vendor-ops") then
        before_link(function (target)
            local torch_dir = os.iorunv("python3", {
                "-c",
                "import torch, os; print(os.path.dirname(torch.__file__))",
            }):trim()
            local torch_lib_dir = path.join(torch_dir, "lib")
            if not os.isdir(torch_lib_dir) then
                raise("Iluvatar vendor operators: torch library directory not found: " .. torch_lib_dir)
            end
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
        end)
    end
target_end()

local iluvatar_arch = get_config("iluvatar-arch") or "ivcore20"
local iluvatar_cuflags = {
    "-Wno-pass-failed",
    "-fPIC",
    "-x", "ivcore",
    "-std=c++17",
    "--cuda-gpu-arch=" .. iluvatar_arch,
}

local function add_iluvatar_cuflags()
    local args = {}
    for _, flag in ipairs(iluvatar_cuflags) do
        table.insert(args, flag)
    end
    table.insert(args, {force = true})
    add_cuflags(table.unpack(args))
end

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
    add_iluvatar_cuflags()
    add_cuflags("-Wno-error=unused-private-field", "-Wno-error=unused-variable", "-Wno-unused-variable")
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
    add_iluvatar_cuflags()
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
        add_iluvatar_cuflags()
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
