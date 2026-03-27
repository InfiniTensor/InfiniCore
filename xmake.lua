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
    -- Required by HIP headers included from torch ATen/hip.
    add_defines("__HIP_PLATFORM_AMD__")
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

option("flash-attn-prebuilt")
    set_default("")
    set_showmenu(true)
    set_description("Path to prebuilt flash_attn .so file or directory containing it. Used for Hygon DCU.")
option_end()

if has_config("aten") then
    add_defines("ENABLE_ATEN")
    local fa_src = get_config("flash-attn")
    local fa_prebuilt = get_config("flash-attn-prebuilt")
    if not fa_prebuilt or fa_prebuilt == "" then
        fa_prebuilt = os.getenv("FLASH_ATTN_PREBUILT")
    end
    if (fa_src and fa_src ~= "") or (fa_prebuilt and fa_prebuilt ~= "") then
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
    end

    if has_config("hygon-dcu") then
        local cuda_sdk = get_config("cuda") or os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
        local dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
        local function normalize_cuda_root(root)
            if not root or root == "" or not os.isdir(root) then
                return nil
            end
            if os.isdir(path.join(root, "include")) then
                return root
            end
            local nested = {
                path.join(root, "cuda"),
                path.join(root, "cuda-12")
            }
            for _, cand in ipairs(nested) do
                if os.isdir(path.join(cand, "include")) then
                    return cand
                end
            end
            return root
        end

        -- Prefer xmake --cuda=... for deterministic SDK include/link paths.
        local normalized_cuda_sdk = normalize_cuda_root(cuda_sdk)
        if normalized_cuda_sdk then
            add_includedirs(path.join(normalized_cuda_sdk, "include"))
            add_linkdirs(path.join(normalized_cuda_sdk, "lib64"))
        end

        -- Keep DTK fallback paths for environments where only DTK_ROOT is set.
        if dtk_root and dtk_root ~= "" and os.isdir(dtk_root) then
            add_includedirs(path.join(dtk_root, "include"))
            add_includedirs(path.join(dtk_root, "cuda", "include"))
            add_linkdirs(path.join(dtk_root, "lib"))
            add_linkdirs(path.join(dtk_root, "cuda", "lib64"))
        end
    end

    on_load(function (target)
        if has_config("aten") then
            -- Hygon DCU: link prebuilt flash_attn BEFORE torch for correct symbol resolution order
            if has_config("hygon-dcu") then
                local fa_prebuilt = get_config("flash-attn-prebuilt")
                if not fa_prebuilt or fa_prebuilt == "" then
                    fa_prebuilt = os.getenv("FLASH_ATTN_PREBUILT")
                end

                local flash_so_dir = nil
                local flash_so_name = nil

                if fa_prebuilt and fa_prebuilt ~= "" then
                    if os.isfile(fa_prebuilt) then
                        flash_so_dir = path.directory(fa_prebuilt)
                        flash_so_name = path.filename(fa_prebuilt)
                    else
                        flash_so_dir = fa_prebuilt
                        local files = os.files(path.join(fa_prebuilt, "flash_attn_2_cuda*.so"))
                        if #files > 0 then
                            flash_so_name = path.filename(files[1])
                        end
                    end
                else
                    local ok, so_path = pcall(function()
                        return os.iorunv("python", {"-c", "import flash_attn_2_cuda; print(flash_attn_2_cuda.__file__)"}):trim()
                    end)
                    if ok and so_path and so_path ~= "" and os.isfile(so_path) then
                        flash_so_dir = path.directory(so_path)
                        flash_so_name = path.filename(so_path)
                    end
                end

                if flash_so_dir and flash_so_name then
                    target:add("linkdirs", flash_so_dir)
                    target:add("ldflags", "-Wl,--no-as-needed", {force = true})
                    target:add("ldflags", "-l:" .. flash_so_name, {force = true})
                    target:add("ldflags", "-Wl,--as-needed", {force = true})
                    print("Flash Attention library: " .. path.join(flash_so_dir, flash_so_name))
                end
            end

            local outdata = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local TORCH_DIR = outdata

            -- Use sysincludedirs (-isystem) so that torch's bundled pybind11 headers
            -- do not shadow the xmake pybind11 package headers.
            target:add(
                "sysincludedirs",
                path.join(TORCH_DIR, "include"),
                path.join(TORCH_DIR, "include/torch/csrc/api/include"),
                { public = true })
            
            target:add(
                "linkdirs",
                path.join(TORCH_DIR, "lib"),
                { public = true }
            )
            local torch_libdir = path.join(TORCH_DIR, "lib")
            target:add("rpathdirs", torch_libdir)
            target:add("ldflags", "-Wl,--no-as-needed", {force = true})
            local torch_links = {"torch", "c10"}
            local function has_torch_lib(name)
                return #os.files(path.join(torch_libdir, "lib" .. name .. ".so*")) > 0
            end
            if has_torch_lib("torch_cuda") then
                table.insert(torch_links, "torch_cuda")
            elseif has_torch_lib("torch_hip") then
                table.insert(torch_links, "torch_hip")
            end
            if has_torch_lib("c10_cuda") then
                table.insert(torch_links, "c10_cuda")
            elseif has_torch_lib("c10_hip") then
                table.insert(torch_links, "c10_hip")
            end
            target:add("links", table.unpack(torch_links), { public = true })
            -- Hard-pin runtime dependency entries to avoid linker dropping HIP torch libs.
            target:add("ldflags", "-L" .. torch_libdir, {force = true})
            if has_torch_lib("torch_hip") then
                target:add("ldflags", "-l:libtorch_hip.so", {force = true})
            end
            if has_torch_lib("c10_hip") then
                target:add("ldflags", "-l:libc10_hip.so", {force = true})
            end
            if has_torch_lib("torch_cuda") then
                target:add("ldflags", "-l:libtorch_cuda.so", {force = true})
            end
            if has_torch_lib("c10_cuda") then
                target:add("ldflags", "-l:libc10_cuda.so", {force = true})
            end
            target:add("ldflags", "-Wl,--as-needed", {force = true})
            print("Torch libraries: " .. table.concat(torch_links, ", "))
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

    after_install(function (target)
        if not has_config("hygon-dcu") then return end
        local fa_prebuilt = get_config("flash-attn-prebuilt")
        if not fa_prebuilt or fa_prebuilt == "" then
            fa_prebuilt = os.getenv("FLASH_ATTN_PREBUILT")
        end

        local flash_so_path = nil
        if fa_prebuilt and fa_prebuilt ~= "" then
            if os.isfile(fa_prebuilt) then
                flash_so_path = fa_prebuilt
            else
                local files = os.files(path.join(fa_prebuilt, "flash_attn_2_cuda*.so"))
                if #files > 0 then flash_so_path = files[1] end
            end
        else
            local ok, so_path = pcall(function()
                return os.iorunv("python", {"-c", "import flash_attn_2_cuda; print(flash_attn_2_cuda.__file__)"}):trim()
            end)
            if ok and so_path and so_path ~= "" and os.isfile(so_path) then
                flash_so_path = so_path
            end
        end

        if flash_so_path then
            local installdir = target:installdir()
            local libdir = path.join(installdir, "lib")
            os.mkdir(libdir)
            os.cp(flash_so_path, libdir)
            print("Copied prebuilt flash_attn library to " .. libdir)
        end
    end)

    after_build(function (target) print(YELLOW .. "[Congratulations!] Now you can install the libraries with \"xmake install\"" .. NC) end)
target_end()

target("_infinicore")
    add_packages("boost")
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
