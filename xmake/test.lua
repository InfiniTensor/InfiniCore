target("infiniutils-test")
    set_kind("binary")
    add_deps("infini-utils")

    set_warnings("all", "error")
    set_languages("cxx17")

    add_files(os.projectdir().."/src/utils-test/*.cc")
    set_installdir(os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini"))
target_end()

target("infiniop-test")
    set_kind("binary")
    add_deps("infini-utils")
    set_default(false)

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_includedirs(INFINI_ROOT.."/include")
    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt")

    if has_config("omp") then
        add_cxxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end

    add_includedirs(os.projectdir().."/src/infiniop-test/include")
    add_files(os.projectdir().."/src/infiniop-test/src/*.cpp")
    add_files(os.projectdir().."/src/infiniop-test/src/ops/*.cpp")

    set_installdir(INFINI_ROOT)
target_end()

target("infinicore-distributed-graph-test")
    set_kind("binary")
    add_deps("infini-utils")
    set_default(false)

    set_warnings("all", "error")
    set_languages("cxx17")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
    add_includedirs(INFINI_ROOT.."/include")
    add_includedirs("include")
    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinirt", "infiniccl", "dl")
    add_files(os.projectdir().."/src/infinicore-distributed-graph-test/*.cpp")

    before_build(function (target)
        local py_inc = os.iorunv("python3-config", {"--includes"}):trim()
        if py_inc and py_inc ~= "" then
            target:add("cxflags", py_inc, {force = true})
        end
    end)

    before_link(function (target)
        local ldflags = os.iorunv("python3-config", {"--ldflags"}):trim()
        if ldflags and ldflags ~= "" then
            target:add("ldflags", ldflags, {force = true})
        end
        target:add("ldflags", "-rdynamic -Wl,--no-as-needed -lpython3.10 -Wl,--unresolved-symbols=ignore-all", {force = true})
    end)

    if not has_config("graph") then
        on_load(function (target)
            raise("infinicore-distributed-graph-test requires --graph=y")
        end)
    end

    set_installdir(INFINI_ROOT)
target_end()

target("infiniccl-test")
    set_kind("binary")
    add_deps("infini-utils")
    set_default(false)

    set_warnings("all", "error")
    set_languages("cxx17")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
    add_includedirs(INFINI_ROOT.."/include")
    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinirt", "infiniccl")
    add_files(os.projectdir().."/src/infiniccl-test/*.cpp")

    set_installdir(os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini"))
target_end()

target("infinirt-test")
    set_kind("binary")
    add_deps("infinirt")
    on_install(function (target) end)

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files(os.projectdir().."/src/infinirt-test/*.cc")
    set_installdir(os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini"))
target_end()

target("infinicore-test")
    set_kind("binary")
    add_deps("infinicore_cpp_api")
    set_default(false)

    set_languages("cxx17")
    set_warnings("all", "error")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
    add_includedirs(INFINI_ROOT.."/include")
    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    -- Add spdlog support
    add_includedirs("third_party/spdlog/include")
    add_defines("SPDLOG_ACTIVE_LEVEL=0")  -- Enable all log levels

    add_files(os.projectdir().."/src/infinicore/*.cc")
    add_files(os.projectdir().."/src/infinicore/context/*.cc")
    add_files(os.projectdir().."/src/infinicore/context/*/*.cc")
    add_files(os.projectdir().."/src/infinicore/tensor/*.cc")
    add_files(os.projectdir().."/src/infinicore/ops/*/*.cc")
    add_files(os.projectdir().."/src/infinicore/nn/*.cc")
    if has_config("mutual-awareness") then
        add_files(os.projectdir().."/src/infinicore/analyzer/*.cc")
    end

    add_files(os.projectdir().."/src/infinicore-test/**.cc")
    set_installdir(INFINI_ROOT)
target_end()
