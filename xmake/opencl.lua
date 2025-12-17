local OPENCL_HEADERS = os.getenv("OPENCL_HEADERS")
local OPENCL_LIB     = os.getenv("OPENCL_LIB")

if not (OPENCL_HEADERS and OPENCL_LIB) then
    raise("Please set OPENCL_HEADERS and OPENCL_LIB environment variables")
end

target("infinirt-opencl")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)
    set_warnings("all", "error")
    set_languages("cxx17")

    on_load(function (target)
        target:add("includedirs", OPENCL_HEADERS)
        target:add("linkdirs", OPENCL_LIB)
        target:add("links", "OpenCL")
    end)

    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end

    add_files("../src/infinirt/opencl/*.cc")
target_end()
