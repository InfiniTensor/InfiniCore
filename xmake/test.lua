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
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end
    
    add_includedirs(os.projectdir().."/src/infiniop-test/include")
    add_files(os.projectdir().."/src/infiniop-test/src/*.cpp")
    add_files(os.projectdir().."/src/infiniop-test/src/ops/*.cpp")
    add_files(os.projectdir().."/test/conv1d_test.cpp")

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

-- target("conv1d_smoke")
--     set_kind("binary")
--     add_deps("infiniop", "infini-utils", "infinirt-nvidia")
--     set_default(false)
--
--     local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
--
--     set_languages("cxx17")
--     set_warnings("all", "error")
--
--     add_includedirs(INFINI_ROOT.."/include")
--     add_linkdirs(INFINI_ROOT.."/lib")
--     add_links("infinirt", "infiniop")
--     add_rpathdirs(INFINI_ROOT.."/lib")
--
--     if has_config("omp") then
--         add_cxflags("-fopenmp")
--         add_ldflags("-fopenmp")
--     end
--
--     add_files(os.projectdir().."/src/infiniop-test/smoke/conv1d_smoke.cpp")
--
--     set_installdir(INFINI_ROOT)
-- target_end()

target("conv1d-test")
    set_kind("binary")
    add_deps("infiniop", "infini-utils")
    set_default(false)

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_includedirs(INFINI_ROOT.."/include")
    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinirt")

    if has_config("omp") then
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end

    add_includedirs(os.projectdir().."/src/utils")

    set_installdir(INFINI_ROOT)
target_end()

