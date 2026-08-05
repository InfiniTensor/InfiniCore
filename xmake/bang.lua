
local NEUWARE_HOME = os.getenv("NEUWARE_HOME") or "/usr/local/neuware"
local FLASH_ATTN_ROOT = get_config("flash-attn")
local INFINI_ROOT = os.getenv("INFINI_ROOT")
    or (os.getenv("HOME") .. "/.infini")

if has_config("aten") then
    add_defines("_GLIBCXX_USE_CXX11_ABI=0")
end

local function cambricon_flash_attn_so_path()
    local explicit = os.getenv("FLASH_ATTN_2_BANG_SO")
    if explicit and explicit ~= "" then
        explicit = explicit:trim()
        if os.isfile(explicit) then
            return explicit
        end
        print(string.format(
            "warning: cambricon+flash-attn: FLASH_ATTN_2_BANG_SO is not a file: %s",
            explicit))
    end
    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        local candidates
            = os.files(path.join(FLASH_ATTN_ROOT, "flash_attn_2_bang*.so"))
        if #candidates > 0 then
            return candidates[1]
        end
    end
    local detected = os.iorunv("python", {
        "-c",
        "import importlib.util; s=importlib.util.find_spec('flash_attn_2_bang'); print(s.origin if s else '')",
    }):trim()
    if detected ~= "" and os.isfile(detected) then
        return detected
    end
    raise("cambricon+flash-attn: flash_attn_2_bang extension not found; pass its site-packages directory via --flash-attn or set FLASH_ATTN_2_BANG_SO")
end

target("flash-attn-cambricon")
    set_kind("phony")
    set_default(false)
    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            print("Cambricon flash-attn extension: "
                  .. cambricon_flash_attn_so_path())
        end)
    end
target_end()

target("infinicore_cpp_api")
    add_rpathdirs(path.join(INFINI_ROOT, "lib"))
    if has_config("aten") then
        on_load(function (target)
            local torch_mlu_dir = os.iorunv("python", {
                "-c",
                "import torch_mlu, os; print(os.path.dirname(torch_mlu.__file__))",
            }):trim()
            target:add("defines", "_GLIBCXX_USE_CXX11_ABI=0")
            target:add(
                "includedirs", path.join(torch_mlu_dir, "csrc"),
                {public = true})
            target:add(
                "linkdirs", path.join(torch_mlu_dir, "csrc", "lib"),
                {public = true})
        end)
        before_link(function (target)
            local torch_dir = os.iorunv("python", {
                "-c",
                "import torch, os; print(os.path.dirname(torch.__file__))",
            }):trim()
            local torch_mlu_dir = os.iorunv("python", {
                "-c",
                "import torch_mlu, os; print(os.path.dirname(torch_mlu.__file__))",
            }):trim()
            local torch_lib = path.join(torch_dir, "lib")
            local torch_mlu_lib = path.join(torch_mlu_dir, "csrc", "lib")
            target:add(
                "shflags",
                "-Wl,--no-as-needed",
                "-L" .. torch_lib,
                "-ltorch_python", "-ltorch_cpu", "-ltorch", "-lc10",
                "-L" .. torch_mlu_lib,
                "-ltorch_mlu", "-ltorch_mlu_bangc",
                "-Wl,--as-needed",
                "-Wl,-rpath," .. torch_lib,
                "-Wl,-rpath," .. torch_mlu_lib,
                {force = true})
            if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
                local flash_so = cambricon_flash_attn_so_path()
                local flash_dir = path.directory(flash_so)
                target:add(
                    "shflags",
                    "-Wl,--no-as-needed",
                    "-L" .. flash_dir,
                    "-l:" .. path.filename(flash_so),
                    "-Wl,--as-needed",
                    "-Wl,-rpath," .. flash_dir,
                    {force = true})
            end
        end)
    end
target_end()

target("_infinicore")
    add_defines("_GLIBCXX_USE_CXX11_ABI=0")
    add_rpathdirs(path.join(INFINI_ROOT, "lib"))
target_end()

add_includedirs(path.join(NEUWARE_HOME, "include"), {public = true})
add_linkdirs(path.join(NEUWARE_HOME, "lib64"))
add_linkdirs(path.join(NEUWARE_HOME, "lib"))
add_links("libcnrt.so")
add_links("libcnnl.so")
add_links("libcnnl_extra.so")
add_links("libcnpapi.so")

rule("mlu")
    set_extensions(".mlu")

    on_load(function (target)
        target:add("includedirs", path.join(os.projectdir(), "include"))
    end)

    on_build_file(function (target, sourcefile)
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))

        local cc = "cncc"

        local includedirs = table.concat(target:get("includedirs"), " ")
        local args = {"-c", sourcefile, "-o", objectfile, "--bang-mlu-arch=mtp_592", "-O3", "-fPIC", "-Wall", "-Werror", "-std=c++17", "-pthread"}

        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end

        os.execv(cc, args)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

local src_dir = path.join(os.projectdir(), "src", "infiniop")

target("infiniop-cambricon")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    add_cxflags("-lstdc++ -fPIC")
    add_cxxflags("-lstdc++ -fPIC")
    set_warnings("all", "error")

    set_languages("cxx17")
    add_files(src_dir.."/devices/bang/*.cc", src_dir.."/ops/*/bang/*.cc")
    local mlu_files = os.files(src_dir .. "/ops/*/bang/*.mlu")
    if #mlu_files > 0 then
        add_files(mlu_files, {rule = "mlu"})
    end
target_end()

target("infinirt-cambricon")
    set_kind("static")
    add_deps("infini-utils")
    set_languages("cxx17")
    on_install(function (target) end)
    -- Add include dirs
    add_files("../src/infinirt/bang/*.cc")
    add_cxflags("-lstdc++ -Wall -Werror -fPIC")
    add_cxxflags("-lstdc++ -Wall -Werror -fPIC")
target_end()

target("infiniccl-cambricon")
    set_kind("static")
    add_deps("infinirt")
    add_deps("infini-utils")
    set_warnings("all", "error")
    set_languages("cxx17")
    on_install(function (target) end)
    
    if has_config("ccl") then
        if is_plat("linux") then
            add_includedirs(NEUWARE_HOME .. "/include")
            add_linkdirs(NEUWARE_HOME .. "/lib64")
            add_links("cncl", "cnrt")

            if has_package("libibverbs") then
                add_links("ibverbs")
                add_defines("CNCL_RDMA_ENABLED=1")
            end

            if is_arch("arm64") then
                add_defines("CNCL_ARM64_COMPAT_MODE=1")
            end

            add_rpathdirs(NEUWARE_HOME .. "/lib64")
            add_runenvs("LD_LIBRARY_PATH", NEUWARE_HOME .. "/lib64")

            add_files("../src/infiniccl/cambricon/*.cc")
            add_cxflags("-fPIC")
            add_cxxflags("-fPIC")
            add_ldflags("-fPIC")
        else
            print("[Warning] CNCL is currently only supported on Linux")
        end
    end
target_end()
