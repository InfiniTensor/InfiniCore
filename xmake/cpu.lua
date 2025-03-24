target("infiniop-cpu")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_warnings("all", "error")
    add_cxflags("-Wno-unknown-pragmas")

    if is_plat("windows") then
        if has_config("omp") then
            add_cxflags("/openmp")
        end
    else
        add_cxflags("-fPIC")
        if has_config("omp") then
            add_cxflags("-fopenmp")
            add_ldflags("-fopenmp")
        end
    end

    set_languages("cxx17")
    add_files("../src/infiniop/devices/cpu/*.cc", "../src/infiniop/ops/*/cpu/*.cc")

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