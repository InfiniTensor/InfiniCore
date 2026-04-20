add_rules("mode.debug", "mode.release")
-- In CI/docker or non-interactive shells, run xmake with -y (e.g. xmake clean -y) to avoid hanging on package prompts.
if is_mode("debug") then
    add_requires("boost", {configs = {stacktrace = true}})
end
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
    -- Ensure CUDA toolkit headers (e.g. cuda_runtime_api.h) are visible to
    -- C++ sources that include ATen CUDA wrappers like CUDAContextLight.h.
    local cuda_dir = get_config("cuda") or os.getenv("CUDA_HOME") or os.getenv("CUDA_ROOT") or "/usr/local/cuda"
    add_includedirs(path.join(cuda_dir, "include"), { public = true })
end

option("cudnn")
    set_default(true)
    set_showmenu(true)
    set_description("Whether to compile cudnn for Nvidia GPU")
option_end()

if has_config("cudnn") then
    add_defines("ENABLE_CUDNN_API")
end

option("cutlass")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile cutlass for Nvidia GPU")
option_end()

if has_config("cutlass") then
    add_defines("ENABLE_CUTLASS_API")
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
    set_default("")
    set_showmenu(true)
    set_description("Path to flash-attention repo. If not set, flash-attention will not used.")
option_end()

if has_config("aten") then
    add_defines("ENABLE_ATEN")
    -- Only enable FlashAttention integration when a non-empty path is provided.
    local flash_attn_cfg = get_config("flash-attn")
    if flash_attn_cfg ~= nil and flash_attn_cfg ~= "" and flash_attn_cfg ~= false then
        add_defines("ENABLE_FLASH_ATTN")
    end
end

-- InfLLM-V2 direct kernels (requires aten; link against infllm_v2 shared library)
--
-- Policy: InfLLM-V2 is optional and must be checked out/built by the user.
-- We do NOT auto-run `git submodule update` or `python setup.py install` from xmake.
--
-- Usage:
--   - auto-detect (if you manually checked out to third_party/infllmv2_cuda_impl):
--       xmake f --aten=y --infllmv2=y
--   - or specify a path (recommended; works without any checkout under this repo):
--       xmake f --aten=y --infllmv2=/abs/path/to/libinfllm_v2.so
--       xmake f --aten=y --infllmv2=/abs/path/to/infllmv2_cuda_impl   # will auto-detect under build/lib.*/
option("infllmv2")
    set_default("")
    set_showmenu(true)
    set_description("Enable InfLLM-V2 support. Value: 'y' (auto-detect under third_party/infllmv2_cuda_impl) or a path to libinfllm_v2.so / infllmv2_cuda_impl root. Requires --aten=y.")
option_end()

local function _infllmv2_enabled()
    local cfg = get_config("infllmv2")
    return cfg ~= nil and cfg ~= "" and cfg ~= false
end

if _infllmv2_enabled() then
    -- Fail fast: C++ code is gated on ENABLE_INFLLMV2 && ENABLE_ATEN.
    if not has_config("aten") then
        error("--infllmv2 requires --aten=y")
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

target("infinicore_cpp_api")
    set_kind("shared")
    add_deps("infiniop", "infinirt", "infiniccl")
    set_languages("cxx17")
    set_symbols("visibility")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    add_includedirs("include")
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    if get_config("flash-attn") ~= "" and get_config("flash-attn") ~= nil then
        add_installfiles("(builddir)/$(plat)/$(arch)/$(mode)/flash-attn*.so", {prefixdir = "lib"})
        if has_config("nv-gpu") then
            add_deps("flash-attn-nvidia")
        end
        if has_config("qy-gpu") then
            add_deps("flash-attn-qy")
        end
    end

    if get_config("flash-attn") and get_config("flash-attn") ~= "" and has_config("qy-gpu") then
        local flash_so_qy = _qy_flash_attn_cuda_so_path()
        local flash_dir_qy = path.directory(flash_so_qy)
        local flash_name_qy = path.filename(flash_so_qy)
        before_link(function (target)
            target:add(
                "shflags",
                "-Wl,--no-as-needed -L" .. flash_dir_qy .. " -l:" .. flash_name_qy .. " -Wl,-rpath," .. flash_dir_qy,
                {force = true}
            )
        end)
    end

    before_build(function (target)
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
            target:add(
                "links",
                "torch",
                "c10",
                "torch_cuda",
                "c10_cuda",
                { public = true }
            )
        end

        -- InfLLM-V2: locate + link infllm_v2 .so
        local resolved_infllmv2 = nil
        if _infllmv2_enabled() then
            local infllmv2_cfg = get_config("infllmv2")

            local function detect_infllmv2_so(infllmv2_root)
                local candidates = os.files(path.join(infllmv2_root, "build", "lib.*", "infllm_v2", "*.so"))
                if candidates and #candidates > 0 then
                    table.sort(candidates)
                    return candidates[1]
                end
                return nil
            end

            local function is_truthy_enable(v)
                if v == true then
                    return true
                end
                if type(v) == "string" then
                    local s = v:lower()
                    return s == "y" or s == "yes" or s == "true" or s == "1" or s == "on"
                end
                return false
            end

            -- 1) If user passed a file path (libinfllm_v2.so / *.so), use it directly.
            if type(infllmv2_cfg) == "string" and infllmv2_cfg ~= "" and os.isfile(infllmv2_cfg) then
                resolved_infllmv2 = infllmv2_cfg
            end

            -- 2) If user passed a directory, try to auto-detect under it.
            if not resolved_infllmv2 and type(infllmv2_cfg) == "string" and infllmv2_cfg ~= "" and os.isdir(infllmv2_cfg) then
                resolved_infllmv2 = detect_infllmv2_so(infllmv2_cfg)
            end

            -- 3) If user passed y/true, try the conventional in-tree location (if present).
            if not resolved_infllmv2 and is_truthy_enable(infllmv2_cfg) then
                local infllmv2_root = path.join(os.projectdir(), "third_party", "infllmv2_cuda_impl")
                if os.isdir(infllmv2_root) then
                    resolved_infllmv2 = detect_infllmv2_so(infllmv2_root)
                end
            end

            if not resolved_infllmv2 then
                local default_root = path.join(os.projectdir(), "third_party", "infllmv2_cuda_impl")
                error(
                    "[InfLLM-V2] Cannot find built InfLLM-V2 shared library (infllm_v2/*.so).\n" ..
                    "You must build it first, then point xmake to it.\n\n" ..
                    "Options:\n" ..
                    "  (A) Pass a direct .so path:\n" ..
                    "      xmake f --aten=y --infllmv2=/abs/path/to/libinfllm_v2.so -cv\n" ..
                    "  (B) Pass an infllmv2_cuda_impl root directory (auto-detects build/lib.*/infllm_v2/*.so):\n" ..
                    "      xmake f --aten=y --infllmv2=/abs/path/to/infllmv2_cuda_impl -cv\n" ..
                    "  (C) If you checked it out under this repo:\n" ..
                    "      " .. default_root .. "\n" ..
                    "      xmake f --aten=y --infllmv2=y -cv\n"
                )
            end

            if has_config("aten") then
                target:add("defines", "ENABLE_INFLLMV2")
            end

            local abs = path.absolute(resolved_infllmv2)
            local so_dir = path.directory(abs)
            local so_name = path.filename(abs)
            -- IMPORTANT: ensure `infinicore_cpp_api` gets a DT_NEEDED on infllm_v2 .so.
            -- Using `shflags` (not `ldflags`) and `--no-as-needed` avoids the linker
            -- dropping the dependency and leaving runtime undefined symbols
            -- (e.g. `mha_varlen_fwd`) that would otherwise require LD_PRELOAD/ctypes preload.
            target:add("shflags", "-Wl,--no-as-needed -L" .. so_dir .. " -l:" .. so_name .. " -Wl,-rpath," .. so_dir, { public = true })
        end
    end)

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
        add_packages("boost")
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

    set_installdir("python/infinicore")
    on_install(function (target)
        -- Make the in-tree Python package usable after `xmake install _infinicore`.
        -- (Reviewer request: keep install logic in install phase, not after_build.)
        local targetfile = target:targetfile()
        if targetfile and os.isfile(targetfile) then
            local libdir = path.join(os.projectdir(), "python", "infinicore", "lib")
            if not os.isdir(libdir) then
                os.mkdir(libdir)
            end
            os.cp(targetfile, path.join(libdir, path.filename(targetfile)))
        end
    end)
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
