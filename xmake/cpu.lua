target("infiniop-cpu")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_warnings("all", "error")

    if is_plat("windows") then
        add_cxflags("/wd4068")
        if has_config("omp") then
            add_cxflags("/openmp")
        end
    else
        if is_arch("x86_64", "i386") then
        -- x86 架构（启用 AVX2/FMA）
        add_cxxflags("-mavx2", "-mfma", "-O3")
        elseif is_arch("arm64", "arm.*") then
            -- ARM 架构（启用 NEON）
            add_cxxflags("-O3", "-mcpu=generic+simd")  -- ARMv8+NEON
        end
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        if has_config("omp") then
            add_cxflags("-fopenmp")
            add_ldflags("-fopenmp")
        end
    end

    set_languages("cxx17")
    add_files("../src/infiniop/devices/cpu/*.cc", "../src/infiniop/ops/*/cpu/*.cc", "../src/infiniop/reduce/cpu/*.cc")

target_end()

target("infinirt-cpu")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_warnings("all", "error")

    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end

    set_languages("cxx17")
    add_files("../src/infinirt/cpu/*.cc")
target_end()

if has_config("omp") then
    add_requires("openmp")
    add_packages("openmp")
end
