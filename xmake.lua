add_rules("mode.debug", "mode.release")
add_requires("boost", {configs = {stacktrace = true}})
add_requires("pybind11")

-- Define color codes
local GREEN = '\27[0;32m'
local YELLOW = '\27[1;33m'
local NC = '\27[0m'  -- No Color

set_encodings("utf-8")

add_includedirs("include")
add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/nlohmann_json/single_include/")

if is_mode("debug") then
    add_defines("DEBUG_MODE")
end

if is_plat("windows") then
    set_runtimes("MD")
    add_ldflags("/utf-8", {force = true})
    add_cxxflags("/utf-8", {force = true})
end

-- CPU
option("cpu")
    set_default(true)
    set_showmenu(true)
    set_description("Whether to compile implementations for CPU")
option_end()

option("omp")
    set_default(true)
    set_showmenu(true)
    set_description("Enable or disable OpenMP support for cpu kernel")
option_end()

if has_config("cpu") then
    includes("xmake/cpu.lua")
    add_defines("ENABLE_CPU_API")
end

if has_config("omp") then
    add_defines("ENABLE_OMP")
end

-- 英伟达
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

option("marlin")
    set_default(false)
    set_showmenu(true)
    set_description("Enable AWQ/GPTQ Marlin GEMM operators (requires --nv-gpu=y; GPTQ Marlin may need TVM_ROOT and --cuda_arch)")
option_end()

if has_config("marlin") then
    add_defines("ENABLE_MARLIN")
end

option("cudnn")
    set_default(true)
    set_showmenu(true)
    set_description("Whether to compile cudnn for Nvidia GPU")
option_end()

if has_config("cudnn") then
    add_defines("ENABLE_CUDNN_API")
end

option("cuda_arch")
    set_showmenu(true)
    set_description("Set CUDA GPU architecture (e.g. sm_90)")
    set_values("sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a")
    set_category("option")
option_end()

-- 寒武纪
option("cambricon-mlu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Cambricon MLU")
option_end()

if has_config("cambricon-mlu") then
    add_defines("ENABLE_CAMBRICON_API")
    includes("xmake/bang.lua")
end

-- 华为昇腾
option("ascend-npu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Huawei Ascend NPU")
option_end()

if has_config("ascend-npu") then
    add_defines("ENABLE_ASCEND_API")
    includes("xmake/ascend.lua")
end

-- 天数智芯
option("iluvatar-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Iluvatar GPU")
option_end()

option("iluvatar_arch")
    set_default("ivcore20")
    set_showmenu(true)
    set_description("Set Iluvatar GPU architecture (e.g. ivcore20)")
    set_values("ivcore20")
    set_category("option")
option_end()

if has_config("iluvatar-gpu") then
    add_defines("ENABLE_ILUVATAR_API")
    includes("xmake/iluvatar.lua")
end

-- ali
option("ali-ppu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Ali PPU")
option_end()

if has_config("ali-ppu") then
    add_defines("ENABLE_ALI_API")
    includes("xmake/ali.lua")
end

-- qy
option("qy-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Qy GPU")
option_end()

if has_config("qy-gpu") then
    add_defines("ENABLE_QY_API")
    includes("xmake/qy.lua")
end

-- 沐曦
option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX GPU")
option_end()

option("use-mc")
    set_default(false)
    set_showmenu(true)
    set_description("Use MC version")
option_end()

if has_config("metax-gpu") then
    add_defines("ENABLE_METAX_API")
    if has_config("use-mc") then
        add_defines("ENABLE_METAX_MC_API")
        -- MACA torch build expects USE_MACA for ATen headers (e.g. C10_WARP_SIZE).
        add_defines("USE_MACA")
    else
        -- HPCC torch build expects this for ATen headers on hpcc.
        add_defines("USE_HPCC")
    end
    includes("xmake/metax.lua")
end

-- 摩尔线程
option("moore-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Moore Threads GPU")
option_end()

option("moore-gpu-arch")
    set_default("mp_31")
    set_showmenu(true)
    set_description("Set Moore GPU architecture (e.g. mp_31)")
option_end()

if has_config("moore-gpu") then
    add_defines("ENABLE_MOORE_API")
    includes("xmake/moore.lua")
end

-- 海光DCU
option("hygon-dcu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Hygon DCU")
option_end()

option("hygon-arch")
    -- gfx906, gfx936
    set_default(nil)
    set_showmenu(true)
    set_description("Set Hygon DCU architecture, e.g. gfx936. If unset, defaults to gfx936 for BW1000 DCU.")
option_end()

if has_config("hygon-dcu") then
    add_defines("ENABLE_HYGON_API")
    includes("xmake/hygon.lua")
end

-- 昆仑芯
option("kunlun-xpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Kunlun XPU kernel")
option_end()

if has_config("kunlun-xpu") then
    add_defines("ENABLE_KUNLUN_API")
    includes("xmake/kunlun.lua")
end

-- 九齿
option("ninetoothed")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to complie NineToothed implementations")
option_end()

if has_config("ninetoothed") then
    add_defines("ENABLE_NINETOOTHED")
end

-- ATen
option("aten")
    set_default(false)
    set_showmenu(true)
    set_description("Wether to link aten and torch libraries")
option_end()

-- Flash-Attn
option("flash-attn")
    set_default(nil)
    set_showmenu(true)
    set_description("Path to flash-attention repo. If not set, flash-attention will not used.")
option_end()

if has_config("aten") then
    add_defines("ENABLE_ATEN")
    if has_config("iluvatar-gpu") then
        add_defines("_GLIBCXX_USE_CXX11_ABI=0")
    end
    if get_config("flash-attn") and get_config("flash-attn") ~= ""
       and (has_config("nv-gpu") or has_config("metax-gpu") or has_config("qy-gpu") or has_config("hygon-dcu")) then
        add_defines("ENABLE_FLASH_ATTN")
    end
end

-- cuda graph
option("graph")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to use device graph instantiating feature, such as cuda graph for nvidia")
option_end()

if has_config("graph") then
    add_defines("USE_INFINIRT_GRAPH")
end


-- InfiniCCL
option("ccl")
    set_default(false)
    set_showmenu(true)
    set_description("Wether to compile implementations for InfiniCCL")
option_end()

if has_config("ccl") then
    add_defines("ENABLE_CCL")
end

-- InfiniOps
option("infiniops")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to use InfiniOps kernels where adapters are available")
option_end()

option("infiniops-root")
    set_default("submodules/InfiniOps")
    set_showmenu(true)
    set_description("Path to the InfiniOps repository used by --infiniops")
option_end()

option("infinirt-root")
    set_default("")
    set_showmenu(true)
    set_description("Path to an installed standalone InfiniRT prefix used by --infiniops")
option_end()

local function get_standalone_infinirt_root()
    local infinirt_root = get_config("infinirt-root")
    if not infinirt_root or infinirt_root == "" then
        infinirt_root = os.getenv("INFINI_RT_ROOT")
    end
    if infinirt_root and infinirt_root ~= "" then
        return path.absolute(infinirt_root, os.projectdir())
    end
    return nil
end

local function get_infiniops_cuda_architectures()
    local arch_opt = get_config("cuda_arch")
    if not arch_opt or arch_opt == "" then
        return nil
    end

    local cmake_archs = {}
    for _, arch in ipairs(arch_opt:gsub(";", ","):split(",")) do
        local cmake_arch = arch:trim():match("^sm_(%d+a?)$")
        if cmake_arch then
            table.insert(cmake_archs, cmake_arch)
        end
    end
    if #cmake_archs == 0 then
        return nil
    end
    return table.concat(cmake_archs, ";")
end

local infiniops_external_built = false

local function filter_infiniops_ops_for_backend(infiniops_ops)
    if not infiniops_ops or #infiniops_ops == 0 then
        return infiniops_ops
    end
    if has_config("nv-gpu") then
        return infiniops_ops
    end

    local skipped_ops = {
        paged_attention_infinilm = true,
        paged_attention_prefill_infinilm = true
    }
    local filtered = {}
    for _, op in ipairs(infiniops_ops:split("[,;]")) do
        op = op:trim()
        if #op > 0 and not skipped_ops[op] then
            table.insert(filtered, op)
        end
    end
    return table.concat(filtered, ",")
end

local function get_infiniops_backend_cmake_arg()
    local enabled = {}
    local function add_backend(config, cmake_arg)
        if has_config(config) then
            table.insert(enabled, cmake_arg)
        end
    end
    add_backend("nv-gpu", "-DWITH_NVIDIA=ON")
    add_backend("metax-gpu", "-DWITH_METAX=ON")
    add_backend("iluvatar-gpu", "-DWITH_ILUVATAR=ON")
    add_backend("moore-gpu", "-DWITH_MOORE=ON")
    if #enabled == 0 then
        raise("InfiniOps integration requires one of --nv-gpu, --metax-gpu, --iluvatar-gpu, or --moore-gpu")
    end
    if #enabled > 1 then
        raise("InfiniOps can build only one GPU backend at a time")
    end
    return enabled[1]
end

local function build_infiniops_external(xmake_os)
    if not has_config("infiniops") or infiniops_external_built then
        return
    end
    local infiniops_root = path.absolute(get_config("infiniops-root") or "submodules/InfiniOps", os.projectdir())
    local infiniops_builddir = path.join(infiniops_root, "build")
    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
    local infinirt_root = get_standalone_infinirt_root()
    local cmake_config_args = {
        "-S", infiniops_root,
        "-B", infiniops_builddir,
        "-DWITH_CPU=ON",
        get_infiniops_backend_cmake_arg(),
        "-DGENERATE_OPERATOR_CALL_INSTANTIATIONS=ON",
        "-DGENERATE_PYTHON_BINDINGS=OFF",
        "-DCMAKE_BUILD_TYPE=Release"
    }
    if has_config("iluvatar-gpu") and has_config("aten") then
        table.insert(cmake_config_args, "-DTORCH_CXX11_ABI=0")
        table.insert(cmake_config_args, "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
    end
    local infiniops_ops = filter_infiniops_ops_for_backend(os.getenv("INFINI_OPS_OPS"))
    if infiniops_ops and #infiniops_ops > 0 then
        table.insert(cmake_config_args, "-DINFINI_OPS_OPS=" .. infiniops_ops)
    end
    local cmake_cuda_architectures = get_infiniops_cuda_architectures()
    if cmake_cuda_architectures and cmake_cuda_architectures ~= "" then
        table.insert(cmake_config_args, "-DCMAKE_CUDA_ARCHITECTURES=" .. cmake_cuda_architectures)
    end
    if infinirt_root and infinirt_root ~= "" then
        table.insert(cmake_config_args, "-DINFINI_RT_ROOT=" .. infinirt_root)
    end
    xmake_os.execv("cmake", cmake_config_args)
    -- The first configure regenerates operator_call_instantiations_*.cc.
    -- Reconfigure once so CMake's globbed infiniops target sees every shard.
    xmake_os.execv("cmake", cmake_config_args)
    xmake_os.execv("cmake", {"--build", infiniops_builddir, "--target", "infiniops"})
    xmake_os.execv("cmake", {"--install", infiniops_builddir, "--prefix", INFINI_ROOT})
    if infinirt_root and infinirt_root ~= "" then
        local standalone_infinirt = path.join(infinirt_root, "lib", "libinfinirt.so")
        if not xmake_os.isfile(standalone_infinirt) then
            standalone_infinirt = path.join(infinirt_root, "lib64", "libinfinirt.so")
        end
        if not xmake_os.isfile(standalone_infinirt) then
            raise("Standalone InfiniRT library not found under: " .. infinirt_root)
        end
        local infiniops_lib = path.join(INFINI_ROOT, "lib", "libinfiniops.so")
        local private_soname = "libinfiniops_infinirt.so"
        local private_infinirt = path.join(INFINI_ROOT, "lib", private_soname)
        xmake_os.cp(standalone_infinirt, private_infinirt)
        xmake_os.execv("patchelf", {"--set-soname", private_soname, private_infinirt})
        xmake_os.execv("patchelf", {"--replace-needed", standalone_infinirt, private_soname, infiniops_lib})
        xmake_os.execv("patchelf", {"--replace-needed", "libinfinirt.so", private_soname, infiniops_lib})
    end
    infiniops_external_built = true
end

-- Mutual Awareness Analyzer
option("mutual-awareness")
    set_default(false)
    set_showmenu(true)
    set_description("Enable hardware-task mutual awareness analyzer and goal-aware kernel dispatch")
option_end()

if has_config("mutual-awareness") then
    add_defines("ENABLE_MUTUAL_AWARENESS")
end

target("infini-utils")
    set_kind("static")
    on_install(function (target) end)
    set_languages("cxx17")

    set_warnings("all", "error")

    if is_plat("windows") then
        add_cxxflags("/wd4068")
        if has_config("omp") then
            add_cxxflags("/openmp")
        end
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cxxflags("-fPIC", "-Wno-unknown-pragmas")
        if has_config("omp") then
            add_cxxflags("-fopenmp")
            add_ldflags("-fopenmp", {force = true})
        end
    end

    add_files("src/utils/*.cc")
target_end()

target("infinirt")
    set_kind("shared")

    if has_config("cpu") then
        add_deps("infinirt-cpu")
    end
    if has_config("nv-gpu") then
        add_deps("infinirt-nvidia")
    end
    if has_config("cambricon-mlu") then
        add_deps("infinirt-cambricon")
    end
    if has_config("ascend-npu") then
        add_deps("infinirt-ascend")
    end
    if has_config("metax-gpu") then
        add_deps("infinirt-metax")
    end
    if has_config("moore-gpu") then
        add_deps("infinirt-moore")
    end
    if has_config("iluvatar-gpu") then
        add_deps("infinirt-iluvatar")
    end
    if has_config("ali-ppu") then
        add_deps("infinirt-ali")
    end
    if has_config("qy-gpu") then
        add_deps("infinirt-qy")
        add_files("build/.objs/infinirt-qy/rules/qy.cuda/src/infinirt/cuda/*.cu.o", {public = true})
    end
    if has_config("kunlun-xpu") then
        add_deps("infinirt-kunlun")
    end
    if has_config("hygon-dcu") then
        add_deps("infinirt-hygon")
    end
    set_languages("cxx17")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")
        add_ldflags("-fPIC", {force = true})
    end
    set_installdir(os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini"))
    add_files("src/infinirt/*.cc")
    add_installfiles("include/infinirt.h", {prefixdir = "include"})
target_end()

target("infiniop")
    set_kind("shared")
    add_deps("infinirt")

    local public_cuda_root = get_config("cuda") or os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
    if public_cuda_root and public_cuda_root ~= "" then
        for _, include_dir in ipairs({
            path.join(public_cuda_root, "include"),
            path.join(public_cuda_root, "targets", "x86_64-linux", "include"),
        }) do
            if os.isdir(include_dir) then
                add_includedirs(include_dir)
            end
        end
        for _, link_dir in ipairs({
            path.join(public_cuda_root, "lib64"),
            path.join(public_cuda_root, "targets", "x86_64-linux", "lib"),
        }) do
            if os.isdir(link_dir) then
                add_linkdirs(link_dir)
            end
        end
    elseif has_config("nv-gpu") then
        add_includedirs(path.join("/usr/local/cuda", "include"))
    end

    if has_config("cpu") then
        add_deps("infiniop-cpu")
    end
    if has_config("nv-gpu") then
        add_deps("infiniop-nvidia")
    end
    if has_config("iluvatar-gpu") then
        add_deps("infiniop-iluvatar")
    end
    if has_config("ali-ppu") then
        add_deps("infiniop-ali")
    end
    if has_config("qy-gpu") then
        add_deps("infiniop-qy")
        add_files("build/.objs/infiniop-qy/rules/qy.cuda/src/infiniop/ops/*/nvidia/*.cu.o", {public = true})
        add_files("build/.objs/infiniop-qy/rules/qy.cuda/src/infiniop/ops/*/*/nvidia/*.cu.o", {public = true})
        add_files("build/.objs/infiniop-qy/rules/qy.cuda/src/infiniop/devices/nvidia/*.cu.o", {public = true})
        add_files("build/.objs/infiniop-qy/rules/qy.cuda/src/infiniop/ops/*/qy/*.cu.o", {public = true})
        add_files("build/.objs/infiniop-qy/rules/qy.cuda/src/infiniop/ops/*/*/qy/*.cu.o", {public = true})
        add_files("build/.objs/infiniop-qy/rules/qy.cuda/src/infiniop/devices/qy/*.cu.o", {public = true})
    end

    if has_config("cambricon-mlu") then
        add_deps("infiniop-cambricon")
    end
    if has_config("ascend-npu") then
        add_deps("infiniop-ascend")
    end
    if has_config("metax-gpu") then
        add_deps("infiniop-metax")
    end
    if has_config("moore-gpu") then
        add_deps("infiniop-moore")
    end
    if has_config("kunlun-xpu") then
        add_deps("infiniop-kunlun")
    end
    if has_config("hygon-dcu") then
        add_deps("infiniop-hygon")
    end
    set_languages("cxx17")
    add_files("src/infiniop/devices/handle.cc")
    add_files("src/infiniop/ops/*/operator.cc", "src/infiniop/ops/*/*/operator.cc")
    if not has_config("marlin") then
        remove_files(
            "src/infiniop/ops/gptq_marlin_gemm/operator.cc",
            "src/infiniop/ops/awq_marlin_gemm/operator.cc"
        )
        add_files("src/infiniop/ops/marlin_stub/operators.cc")
    end
    add_files("src/infiniop/*.cc")

    set_installdir(os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini"))
    add_installfiles("include/infiniop/(**/*.h)", {prefixdir = "include/infiniop"})
    add_installfiles("include/infiniop/*.h", {prefixdir = "include/infiniop"})
    add_installfiles("include/infiniop.h", {prefixdir = "include"})
    add_installfiles("include/infinicore.h", {prefixdir = "include"})
target_end()

target("infiniccl")
    set_kind("shared")
    add_deps("infinirt")

    if has_config("nv-gpu") then
        add_deps("infiniccl-nvidia")
    end
    if has_config("ascend-npu") then
        add_deps("infiniccl-ascend")
    end
    if has_config("cambricon-mlu") then
        add_deps("infiniccl-cambricon")
    end
    if has_config("metax-gpu") then
        add_deps("infiniccl-metax")
    end
    if has_config("iluvatar-gpu") then
        add_deps("infiniccl-iluvatar")
    end
    if has_config("ali-ppu") then
        add_deps("infiniccl-ali")
    end
    if has_config("qy-gpu") then
        add_deps("infiniccl-qy")
        add_files("build/.objs/infiniccl-qy/rules/qy.cuda/src/infiniccl/cuda/*.cu.o", {public = true})
    end

    if has_config("moore-gpu") then
        add_deps("infiniccl-moore")
    end

    if has_config("kunlun-xpu") then
        add_deps("infiniccl-kunlun")
    end
    if has_config("hygon-dcu") then
        add_deps("infiniccl-hygon")
    end

    set_languages("cxx17")

    add_files("src/infiniccl/*.cc")
    add_installfiles("include/infiniccl.h", {prefixdir = "include"})

    set_installdir(os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini"))
target_end()

target("infinicore_c_api")
    set_kind("phony")
    add_deps("infiniop", "infinirt", "infiniccl")
    after_build(function (target) print(YELLOW .. "[Congratulations!] Now you can install the libraries with \"xmake install\"" .. NC) end)
target_end()

target("infiniops_external")
    set_kind("phony")
    set_default(false)

    on_build(function (target)
        build_infiniops_external(os)
    end)
target_end()

target("infinicore_cpp_api")
    set_kind("shared")
    add_deps("infiniop", "infinirt", "infiniccl")
    set_languages("cxx17")
    set_symbols("visibility")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    add_includedirs("include")
    if has_config("metax-gpu") and has_config("use-mc") and has_config("aten") then
        local maca_root = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or os.getenv("MACA_ROOT") or "/opt/maca"
        add_includedirs(maca_root .. "/include")
        add_includedirs(maca_root .. "/tools/cu-bridge/include")
        for _, include_dir in ipairs(os.dirs(maca_root .. "/include/*")) do
            add_includedirs(include_dir)
        end
        add_cxflags("-include", "climits")
        add_cxxflags("-includeclimits", {force = true})
        add_defines("CHAR_BIT=8", "INT_MIN=(-2147483647 - 1)", "INT_MAX=2147483647", "UINT_MAX=4294967295U")
    end
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    if has_config("nv-gpu") then
        local cuda_root = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or get_config("cuda") or "/usr/local/cuda"
        add_includedirs(cuda_root .. "/include")
    end
    if has_config("infiniops") then
        local infiniops_root = path.absolute(get_config("infiniops-root") or "submodules/InfiniOps", os.projectdir())
        if not os.isdir(infiniops_root) then
            raise("InfiniOps root not found: " .. infiniops_root)
        end
        get_infiniops_backend_cmake_arg()
        local infinirt_root = get_standalone_infinirt_root()
        if infinirt_root and infinirt_root ~= "" then
            add_includedirs(infinirt_root .. "/include")
            add_linkdirs(infinirt_root .. "/lib", infinirt_root .. "/lib64")
            add_rpathdirs(infinirt_root .. "/lib", infinirt_root .. "/lib64")
        end
        add_deps("infiniops_external")
        add_defines("ENABLE_INFINIOPS_API")
        add_links("infiniops")
        add_rpathdirs(INFINI_ROOT .. "/lib")
        on_load(function (target)
            build_infiniops_external(os)
        end)
        after_install(function (target)
            local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
            local infinirt_root = get_standalone_infinirt_root()
            if infinirt_root and infinirt_root ~= "" then
                local standalone_infinirt = path.join(infinirt_root, "lib", "libinfinirt.so")
                if not os.isfile(standalone_infinirt) then
                    standalone_infinirt = path.join(infinirt_root, "lib64", "libinfinirt.so")
                end
                if not os.isfile(standalone_infinirt) then
                    raise("Standalone InfiniRT library not found under: " .. infinirt_root)
                end
                local infiniops_lib = path.join(INFINI_ROOT, "lib", "libinfiniops.so")
                local private_soname = "libinfiniops_infinirt.so"
                local private_infinirt = path.join(INFINI_ROOT, "lib", private_soname)
                os.cp(standalone_infinirt, private_infinirt)
                os.execv("patchelf", {"--set-soname", private_soname, private_infinirt})
                os.execv("patchelf", {"--replace-needed", "libinfinirt.so", private_soname, infiniops_lib})
            end
        end)
    end

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    if get_config("flash-attn") and get_config("flash-attn") ~= "" then
        add_installfiles("(builddir)/$(plat)/$(arch)/$(mode)/flash-attn*.so", {prefixdir = "lib"})
        if has_config("nv-gpu") then
            add_deps("flash-attn-nvidia")
        end
        if has_config("metax-gpu") then
            add_deps("flash-attn-metax")
        end
        if has_config("qy-gpu") then
            add_deps("flash-attn-qy")
        end
        if has_config("hygon-dcu") then
            add_deps("flash-attn-hygon")
        end
    end

    -- Flash pip `.so` link flags: `before_link` runs in an xmake sandbox that cannot see helpers
    -- from other included scripts; MetaX and QY each register their own hook in `xmake/metax.lua`
    -- and `xmake/qy.lua`.

    -- Moore mate: 
    -- enable Python bridge macro for flash-attn Moore path
    -- pybind11/embed.h for mha_kvcache branch
    if has_config("moore-gpu") and has_config("aten") and has_config("flash-attn") then
        add_defines("ENABLE_MOORE_MATE_FLASH_ATTN")
        add_packages("pybind11")
    end

    before_build(function (target)
        -- MetaX + flash-attn: `flash_attn_2_cuda` may use a different `mha_fwd_kvcache` ABI
        -- depending on the underlying stack version. When building with MACA (`--use-mc=y`),
        -- the version file is typically `/opt/maca/Version.txt` (HPCC uses `/opt/hpcc/Version.txt`).
        if has_config("metax-gpu") and get_config("flash-attn") and get_config("flash-attn") ~= "" then
            local version_txt = "/opt/hpcc/Version.txt"
            if not os.isfile(version_txt) and has_config("use-mc") then
                version_txt = "/opt/maca/Version.txt"
            end
            if os.isfile(version_txt) then
                local content = os.iorunv("cat", {version_txt}) or ""
                content = content:trim()
                local major_str = content:match("Version:(%d+)") or content:match("^(%d+)")
                if major_str and major_str ~= "" then
                    local major = tonumber(major_str)
                    if major then
                        local define = "INFINICORE_HPCC_VERSION_MAJOR=" .. tostring(major)
                        target:add("defines", define)
                        target:add("cxflags", "-D" .. define)
                        target:add("cxxflags", "-D" .. define)
                    end
                end
            end
        end

        if has_config("aten") then
            local outdata = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local TORCH_DIR = outdata

            target:add(
                "includedirs", 
                path.join(TORCH_DIR, "include"), 
                path.join(TORCH_DIR, "include/torch/csrc/api/include"),
                { public = true })
            
            target:add(
                "linkdirs",
                path.join(TORCH_DIR, "lib"),
                { public = true }
            )

            -- Moore mate: link torch_musa instead of torch_cuda/c10_cuda
            if has_config("moore-gpu") then
                target:add(
                    "links",
                    "torch",
                    "torch_cpu",
                    "torch_python",
                    "c10",
                    { public = true }
                )

                -- Detect torch_musa install path
                local musa_outdata = os.iorunv("python", {"-c", "import torch_musa, os; print(os.path.dirname(torch_musa.__file__))"}):trim()
                local TORCH_MUSA_DIR = musa_outdata
                local MUSA_ROOT = os.getenv("MUSA_ROOT") or os.getenv("MUSA_HOME") or os.getenv("MUSA_PATH") or "/usr/local/musa"

                target:add(
                    "includedirs",
                    path.join(MUSA_ROOT, "include"),
                    path.directory(TORCH_MUSA_DIR),
                    path.join(TORCH_MUSA_DIR, "include"),
                    path.join(TORCH_MUSA_DIR, "share/generated_cuda_compatible/include"),
                    path.join(TORCH_MUSA_DIR, "share/generated_cuda_compatible"),
                    { public = true }
                )

                target:add(
                    "linkdirs",
                    path.join(TORCH_MUSA_DIR, "lib"),
                    { public = true }
                )
                target:add(
                    "links",
                    "musa_python",
                    "musa_kernels",
                    { public = true }
                )

                -- libpython for pybind11::scoped_interpreter / embed
                local pyinc = os.iorunv("python", {"-c",
                    "import sysconfig; print(sysconfig.get_path('include'))"}):trim()
                local pylib = os.iorunv("python", {"-c",
                    "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"}):trim()
                local pyver = os.iorunv("python", {"-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"}):trim()
                target:add("includedirs", pyinc, { public = true })
                target:add("linkdirs",    pylib, { public = true })
                target:add("links",       "python" .. pyver, { public = true })

                target:add(
                    "shflags",
                    "-Wl,-rpath," .. path.join(TORCH_MUSA_DIR, "lib"),
                    "-Wl,-rpath," .. path.join(MUSA_ROOT, "lib"),
                    "-Wl,-rpath," .. path.join(TORCH_DIR, "lib"),
                    "-Wl,-rpath," .. pylib,
                    { force = true }
                )
            elseif has_config("hygon-dcu") then
                target:add(
                    "links",
                    "torch",
                    "torch_cpu",
                    "torch_python",
                    "c10",
                    "torch_hip",
                    "c10_hip",
                    { public = true }
                )
            else
                target:add(
                    "links",
                    "torch",
                    "c10",
                    "torch_cuda",
                    "c10_cuda",
                    { public = true }
                )
            end
        end

    end)

    -- Moore mate: force link torch_python to bypass --as-needed
    if has_config("moore-gpu") and has_config("aten") and has_config("flash-attn") then
        before_link(function (target)
            local torch_dir = os.iorunv("python", {"-c",
                "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local torch_lib = path.join(torch_dir, "lib")
            target:add("shflags",
                "-Wl,--no-as-needed",
                "-L" .. torch_lib,
                "-ltorch_python",
                "-ltorch_cpu",
                "-lc10",
                "-Wl,--as-needed",
                "-Wl,-rpath," .. torch_lib,
                {force = true})
        end)
    end

    -- Add InfiniCore C++ source files (needed for RoPE and other nn modules)
    add_files("src/infinicore/*.cc")
    add_files("src/infinicore/adaptor/*.cc")
    add_files("src/infinicore/context/*.cc")
    add_files("src/infinicore/context/*/*.cc")
    add_files("src/infinicore/tensor/*.cc")
    add_files("src/infinicore/graph/*.cc")
    add_files("src/infinicore/nn/*.cc")
    add_files("src/infinicore/ops/*/*.cc")
    add_files("src/infinicore/ops/*/*/*.cc")
    -- Platform-private Hygon sources are guarded and only kept in Hygon builds.
    if has_config("hygon-dcu") and get_config("flash-attn") and get_config("flash-attn") ~= "" then
        -- Hygon links against a prebuilt flash-attn extension with a different ABI.
        -- Keep the platform-specific implementation under each op's hygon/ folder.
        add_files("src/infinicore/adaptor/flash_attn/hygon/*.cc")
        remove_files("src/infinicore/ops/multi_head_attention/mha_flashattn.cc")
        remove_files("src/infinicore/ops/multi_head_attention_varlen/mha_varlen_flashattn.cc")
        remove_files("src/infinicore/ops/mha_kvcache/mha_kvcache_flashattn.cc")
    else
        remove_files("src/infinicore/ops/*/hygon/*.cc")
    end
    if has_config("infiniops") and not has_config("nv-gpu") then
        remove_files("src/infinicore/ops/paged_attention/paged_attention_infiniops.cc")
        remove_files("src/infinicore/ops/paged_attention_prefill/paged_attention_prefill_infiniops.cc")
    end
    
    if not has_config("marlin") then
        remove_files(
            "src/infinicore/ops/awq_marlin_gemm/*.cc",
            "src/infinicore/ops/gptq_marlin_gemm/*.cc"
        )
    end
    if has_config("mutual-awareness") then
        add_files("src/infinicore/analyzer/*.cc")
    end
    add_files("src/utils/*.cc")

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore/(**.h)",    {prefixdir = "include/infinicore"})
    add_installfiles("include/infinicore/(**.hpp)",    {prefixdir = "include/infinicore"})
    add_installfiles("include/infinicore/(**/*.h)",  {prefixdir = "include/infinicore"})
    add_installfiles("include/infinicore/(**/*.hpp)",{prefixdir = "include/infinicore"})
    add_installfiles("include/infinicore.h",          {prefixdir = "include"})
    add_installfiles("include/infinicore.hpp",        {prefixdir = "include"})
    after_build(function (target) print(YELLOW .. "[Congratulations!] Now you can install the libraries with \"xmake install\"" .. NC) end)
target_end()

target("_infinicore")
    if is_mode("debug") then
        add_defines("BOOST_STACKTRACE_USE_BACKTRACE")
        add_links("backtrace")
    else
        add_defines("BOOST_STACKTRACE_USE_NOOP")
    end

    set_default(false)
    add_rules("python.library", {soabi = true})
    add_packages("pybind11")
    set_languages("cxx17")

    add_deps("infinicore_cpp_api")

    set_kind("shared")
    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    add_files("src/infinicore/pybind11/**.cc")

    if has_config("infiniops") then
        local infinirt_root = get_standalone_infinirt_root()
        if infinirt_root and infinirt_root ~= "" then
            add_includedirs(infinirt_root .. "/include")
            add_linkdirs(infinirt_root .. "/lib", infinirt_root .. "/lib64")
            add_rpathdirs(infinirt_root .. "/lib", infinirt_root .. "/lib64")
        end
        after_install(function (target)
            local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
            local infiniops_root = path.absolute(get_config("infiniops-root") or "submodules/InfiniOps", os.projectdir())
            local infiniops_lib = path.join(INFINI_ROOT, "lib", "libinfiniops.so")
            local infiniops_lib_installed = true
            if not os.isfile(infiniops_lib) then
                infiniops_lib = path.join(infiniops_root, "build", "src", "libinfiniops.so")
                infiniops_lib_installed = false
            end
            local infinirt_root = get_standalone_infinirt_root()
            if infinirt_root and infinirt_root ~= "" then
                local standalone_infinirt = path.join(infinirt_root, "lib", "libinfinirt.so")
                if not os.isfile(standalone_infinirt) then
                    standalone_infinirt = path.join(infinirt_root, "lib64", "libinfinirt.so")
                end
                if not os.isfile(standalone_infinirt) then
                    raise("Standalone InfiniRT library not found under: " .. infinirt_root)
                end
                local private_soname = "libinfiniops_infinirt.so"
                local private_infinirt = path.join(INFINI_ROOT, "lib", private_soname)
                os.cp(standalone_infinirt, private_infinirt)
                os.execv("patchelf", {"--set-soname", private_soname, private_infinirt})
                os.execv("patchelf", {"--replace-needed", standalone_infinirt, private_soname, infiniops_lib})
                os.execv("patchelf", {"--replace-needed", "libinfinirt.so", private_soname, infiniops_lib})
            end
            os.mkdir(path.join(INFINI_ROOT, "lib"))
            if not infiniops_lib_installed then
                os.cp(infiniops_lib, path.join(INFINI_ROOT, "lib"))
            end
            os.mkdir(path.join(os.projectdir(), "python", "infinicore", "lib"))
            os.cp(infiniops_lib, path.join(os.projectdir(), "python", "infinicore", "lib"))
            local private_infinirt = path.join(INFINI_ROOT, "lib", "libinfiniops_infinirt.so")
            if os.isfile(private_infinirt) then
                os.cp(private_infinirt, path.join(os.projectdir(), "python", "infinicore", "lib"))
            end
        end)
    end

    set_installdir("python/infinicore")
target_end()

option("editable")
    set_default(false)
    set_showmenu(true)
    set_description("Install the `infinicore` Python package in editable mode")
option_end()

target("infinicore")
    set_kind("phony")

    set_default(false)

    add_deps("_infinicore")

    on_install(function (target)
        local pip_install_args = {}

        if has_config("editable") then
            table.insert(pip_install_args, "--editable")
        end

        os.execv("python", table.join({"-m", "pip", "install"}, pip_install_args, {"."}))
    end)
target_end()

-- Tests
includes("xmake/test.lua")
