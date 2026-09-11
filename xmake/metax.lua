
local MACA_ROOT = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or os.getenv("MACA_ROOT")
local FLASH_ATTN_ROOT = get_config("flash-attn")

-- MetaX flash-attn (pip `flash_attn_2_cuda`) ships two incompatible forward ABIs
-- (see include/infinicore/adaptor/flash_attention_adaptor.hpp):
--   253: flash_attn 2.5.3 wheels (MACA/HPCC 2.x) -- mha_fwd/mha_varlen_fwd/mha_fwd_kvcache take 13/18/18 args
--   263: flash_attn 2.6.3+metax wheels (MACA/HPCC 3.x) -- the same functions take 16/23/21 args
-- The wheel `.so` that will actually be linked is the ground truth, so its dynamic symbols
-- are inspected at load time (same approach as Cambricon in `xmake/bang.lua`); the HPCC/MACA
-- toolkit Version.txt is only a fallback when the wheel cannot be inspected.

-- Resolve MetaX flash-attn .so path (used only from this file: `before_link` sandbox cannot see globals from `xmake.lua`).
local FLASH_ATTN_METAX_CUDA_SO_CONTAINER_DEFAULT =
    "/opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so"

local function metax_flash_attn_cuda_so_path()
    -- Highest priority: override the exact `.so` file to link.
    local env_path = os.getenv("FLASH_ATTN_2_CUDA_SO")
    if env_path and env_path ~= "" then
        env_path = env_path:trim()
        if os.isfile(env_path) then
            return env_path
        end
        print(string.format("warning: metax+flash-attn: FLASH_ATTN_2_CUDA_SO is not a file: %s, fallback to container/default path", env_path))
    end

    -- Second priority: allow overriding the "expected" container path via env.
    local container_path = os.getenv("FLASH_ATTN_METAX_CUDA_SO_CONTAINER")
    if not container_path or container_path == "" then
        container_path = FLASH_ATTN_METAX_CUDA_SO_CONTAINER_DEFAULT
    end

    if not os.isfile(container_path) then
        print(
            string.format(
                "warning: metax+flash-attn: expected %s; install flash-attn in conda env, or export FLASH_ATTN_2_CUDA_SO.",
                container_path
            )
        )
    end
    return container_path
end

-- Classify a MetaX flash-attn wheel as "253"/"263" from its demangled dynamic symbols.
-- Returns nil + reason when the wheel cannot be inspected or recognized.
-- `run(program, argv)` is injected by the caller: xmake (>= 3.x) restricts the script-body
-- sandbox (no os.iorunv/pcall/try there), while hook functions get the full sandbox, so the
-- on_load hook below passes a runner built from its own environment into these body-level
-- helpers. The runner returns nil instead of raising when the program is unavailable.
local function metax_detect_flash_attn_abi(so_path, run)
    if not so_path or not os.isfile(so_path) then
        return nil, "wheel .so not found"
    end
    local symbols = run("nm", {"-D", "-C", "--defined-only", so_path})
    if not symbols or symbols == "" then
        return nil, "could not read the dynamic symbols of " .. so_path
    end
    if not symbols:find("mha_fwd(at::Tensor&", 1, true) then
        return nil, "no demangled mha_fwd symbol in " .. so_path
    end
    -- Both markers exist only in flash_attn 2.6.3+metax wheels:
    --   * leftpad_k (`optional<at::Tensor const>&`) in mha_varlen_fwd -- primary evidence,
    --     it is one of the params the 2.6.3 ABI appends (substring matches the
    --     `std::optional` and `c10::optional` demangled spellings alike);
    --   * mha_fwd_kvcache_dequant -- corroboration.
    local varlen_sig = symbols:match("[^\n]*mha_varlen_fwd%(([^\n]*)") or ""
    local has_leftpad = varlen_sig:find("optional<at::Tensor const>&", 1, true) ~= nil
    local has_dequant = symbols:find("mha_fwd_kvcache_dequant(", 1, true) ~= nil
    if has_leftpad ~= has_dequant then
        print(string.format(
            "warning: metax+flash-attn: inconsistent ABI markers in %s (varlen leftpad_k=%s, mha_fwd_kvcache_dequant=%s); trusting leftpad_k",
            so_path, tostring(has_leftpad), tostring(has_dequant)))
    end
    if has_leftpad then
        return "263"
    end
    return "253"
end

-- Legacy fallback: HPCC (`/opt/hpcc/Version.txt`) or MACA (`/opt/maca/Version.txt`, with
-- `--use-mc=y`) toolkit major version. MACA/HPCC 3.x stacks ship flash_attn 2.6.3+metax.
local function metax_stack_version_major(run)
    local version_txt = "/opt/hpcc/Version.txt"
    if not os.isfile(version_txt) and has_config("use-mc") then
        version_txt = "/opt/maca/Version.txt"
    end
    if not os.isfile(version_txt) then
        return nil
    end
    local content = run("cat", {version_txt}) or ""
    content = content:trim()
    local major_str = content:match("Version:(%d+)") or content:match("^(%d+)")
    if major_str and major_str ~= "" then
        return tonumber(major_str)
    end
    return nil
end

-- MetaX flash-attn ABI selection + link flags for pip `flash_attn_2_cuda`.
-- `INFINICORE_METAX_FA_ABI` is added {public = true} so it also reaches `infinicore-test`,
-- which depends on this target and compiles the same `mha_*_flashattn.cc` sources.
target("infinicore_cpp_api")
    if get_config("flash-attn") and get_config("flash-attn") ~= "" then
        on_load(function (target)
            -- This hook body runs in xmake's full sandbox (unlike the restricted
            -- script-body scope), so os.iorunv/import are available here.
            local find_program = import("lib.detect.find_program")
            local function run(program, argv)
                if not find_program(program) then
                    return nil
                end
                return os.iorunv(program, argv)
            end
            local abi = get_config("metax-fa-abi")
            if not abi or abi == "" or abi == "auto" then
                local so_path = metax_flash_attn_cuda_so_path()
                local detected, why = metax_detect_flash_attn_abi(so_path, run)
                if detected then
                    abi = detected
                    print(string.format("metax+flash-attn: %s ABI detected from wheel symbols: %s", abi, so_path))
                else
                    local major = metax_stack_version_major(run)
                    if major then
                        -- Header derives the ABI from the toolkit major version (>= 3 -> 263).
                        target:add("defines", "INFINICORE_HPCC_VERSION_MAJOR=" .. tostring(major), {public = true})
                        print(string.format(
                            "metax+flash-attn: could not inspect the wheel (%s); falling back to HPCC/MACA major version %d",
                            why or "unknown reason", major))
                    else
                        print(string.format(
                            "warning: metax+flash-attn: could not inspect the wheel (%s) and found no HPCC/MACA Version.txt; defaulting to the flash_attn 2.5.3 ABI",
                            why or "unknown reason"))
                    end
                end
            end
            if abi == "253" or abi == "263" then
                target:add("defines", "INFINICORE_METAX_FA_ABI=" .. abi, {public = true})
            end
        end)
        before_link(function (target)
            local flash_so_metax = metax_flash_attn_cuda_so_path()
            local flash_dir_metax = path.directory(flash_so_metax)
            local flash_name_metax = path.filename(flash_so_metax)
            target:add(
                "shflags",
                "-Wl,--no-as-needed -L" .. flash_dir_metax .. " -l:" .. flash_name_metax .. " -Wl,-rpath," .. flash_dir_metax,
                {force = true}
            )
        end)
    end
target_end()

add_includedirs(MACA_ROOT .. "/include")
add_linkdirs(MACA_ROOT .. "/lib")
if has_config("use-mc") then
    add_links("mcdnn", "mcblas", "mcruntime")
else
    add_links("hcdnn", "hcblas", "hcruntime")
end

rule("maca")
    set_extensions(".maca")

    on_load(function (target)
        target:add("includedirs", "include")
    end)

    on_build_file(function (target, sourcefile)
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))
        local args
        local htcc
        if has_config("use-mc") then
            htcc = path.join(MACA_ROOT, "mxgpu_llvm/bin/mxcc")
            args = { "-x", "maca", "-c", sourcefile, "-o", objectfile, "-I" .. MACA_ROOT .. "/include", "-O3", "-fPIC", "-Werror", "-std=c++17"}
        else
            htcc = path.join(MACA_ROOT, "htgpu_llvm/bin/htcc")
            args = { "-x", "hpcc", "-c", sourcefile, "-o", objectfile, "-I" .. MACA_ROOT .. "/include", "-O3", "-fPIC", "-Werror", "-std=c++17"}
        end
        local includedirs = table.concat(target:get("includedirs"), " ")
        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end

        local defines = target:get("defines")
        for _, define in ipairs(defines) do
            table.insert(args, "-D" .. define)
        end

        os.execv(htcc, args)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("infiniop-metax")
    set_kind("static")
    on_install(function (target) end)
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cxflags("-lstdc++", "-fPIC", "-Wno-defaulted-function-deleted", "-Wno-strict-aliasing", {force = true})
    add_cxxflags("-lstdc++", "-fPIC", "-Wno-defaulted-function-deleted", "-Wno-strict-aliasing", {force = true})
    add_files("../src/infiniop/devices/metax/*.cc", "../src/infiniop/ops/*/metax/*.cc")
    add_files("../src/infiniop/ops/*/metax/*.maca", {rule = "maca"})

    if has_config("ninetoothed") then
        add_includedirs(MACA_ROOT .. "/include/hcr")
        add_includedirs(MACA_ROOT .. "/include/mcr")
        add_files("../build/ninetoothed/*.c", "../build/ninetoothed/*.cpp", {
            cxflags = {
                "-include stdlib.h",
                "-Wno-return-type",
                "-Wno-implicit-function-declaration",
                "-Wno-builtin-declaration-mismatch"
            }
        })
    end
target_end()

target("flash-attn-metax")
    set_kind("phony")
    set_default(false)

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            local TORCH_DIR = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local PYTHON_INCLUDE = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_paths()['include'])"}):trim()
            local PYTHON_LIB_DIR = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"}):trim()

            -- Validate build/runtime env in container and keep these paths available for downstream linking.
            target:add("includedirs", TORCH_DIR .. "/include", TORCH_DIR .. "/include/torch/csrc/api/include", PYTHON_INCLUDE, {public = false})
            target:add("linkdirs", TORCH_DIR .. "/lib", PYTHON_LIB_DIR, {public = false})
        end)
    else
        before_build(function (target)
            print("Flash Attention not available, skipping flash-attn-metax integration")
        end)
    end
target_end()

target("infinirt-metax")
    set_kind("static")
    set_languages("cxx17")
    on_install(function (target) end)
    add_deps("infini-utils")
    set_warnings("all", "error")
    add_cxflags("-lstdc++ -fPIC")
    add_cxxflags("-lstdc++ -fPIC")
    add_files("../src/infinirt/metax/*.cc")
target_end()

target("infiniccl-metax")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")
    end
    if has_config("ccl") then
        if has_config("use-mc") then
            add_links("libmccl.so")
        else
            add_links("libhccl.so")
        end
        add_files("../src/infiniccl/metax/*.cc")
    end
    set_languages("cxx17")

target_end()
