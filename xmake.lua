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

if is_mode("debug") then
    add_defines("DEBUG_MODE")
end

if is_plat("windows") then
    set_runtimes("MD")
    add_ldflags("/utf-8", {force = true})
    add_cxflags("/utf-8", {force = true})
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

if has_config("iluvatar-gpu") then
    add_defines("ENABLE_ILUVATAR_API")
    includes("xmake/iluvatar.lua")
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
        add_cxflags("/wd4068")
        if has_config("omp") then
            add_cxflags("/openmp")
        end
    else
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        if has_config("omp") then
            add_cxflags("-fopenmp")
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
    if has_config("kunlun-xpu") then
        add_deps("infinirt-kunlun")
    end
    if has_config("hygon-dcu") then
        add_deps("infinirt-hygon")
    end
    set_languages("cxx17")
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
    add_files("src/infiniop/ops/*/operator.cc")
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

    set_kind("shared")
    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    add_files("src/infinicore/*.cc")
    add_files("src/infinicore/context/*.cc")
    add_files("src/infinicore/context/*/*.cc")
    add_files("src/infinicore/tensor/*.cc")
    add_files("src/infinicore/ops/*/*.cc")
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
