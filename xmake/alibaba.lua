toolchain("ali.toolchain")
    set_toolset("cc"  , "clang"  )
    set_toolset("cxx" , "clang++")
    set_toolset("cu"  , "clang++")
    set_toolset("culd", "clang++")
    set_toolset("cu-ccbin", "$(env CXX)", "$(env CC)")
toolchain_end()

rule("ali.env")
    add_deps("cuda.env", {order = true})
    after_load(function (target)
        local old = target:get("syslinks")
        local new = {}

        for _, link in ipairs(old) do
            if link ~= "cudadevrt" then
                table.insert(new, link)
            end
        end

        if #old > #new then
            target:set("syslinks", new)
            local log = "cudadevrt removed, syslinks = { "
            for _, link in ipairs(new) do
                log = log .. link .. ", "
            end
            log = log:sub(0, -3) .. " }"
            print(log)
        end
    end)
rule_end()

target("infiniop-ali")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("ali.toolchain")
    add_rules("ali.env")
    set_values("cuda.rdc", false)

    add_links("cudart", "cublas", "cudnn")

    set_warnings("all", "error")
    add_cuflags("-Wno-error=unused-private-field")
    add_cuflags("-fPIC", "-x", "ivcore", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")
    add_cxxflags("-fPIC")

    set_languages("cxx17")
    add_files("../src/infiniop/devices/nvidia/*.cu", "../src/infiniop/ops/*/nvidia/*.cu")

    add_files("../src/infiniop/ops/dequantize_awq/ali/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", "../build/ninetoothed/*.cpp", {cxxflags = {"-Wno-return-type"}})
    end
target_end()

target("infinirt-ali")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("ali.toolchain")
    add_rules("ali.env")
    set_values("cuda.rdc", false)

    add_links("cudart")

    set_warnings("all", "error")
    add_cuflags("-fPIC", "-x", "ivcore", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")
    add_cxxflags("-fPIC")

    set_languages("cxx17")
    add_files("../src/infinirt/cuda/*.cu")
target_end()

target("infiniccl-ali")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)

    if has_config("ccl") then
        set_toolchains("ali.toolchain")
        add_rules("ali.env")
        set_values("cuda.rdc", false)

        add_links("cudart")

        set_warnings("all", "error")
        add_cuflags("-fPIC", "-x", "ivcore", "-std=c++17", {force = true})
        add_culdflags("-fPIC")
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")

        local nccl_root = os.getenv("NCCL_ROOT")
        if nccl_root then
            add_includedirs(nccl_root .. "/include")
            add_links(nccl_root .. "/lib/libnccl.so")
        else
            add_links("nccl") -- Fall back to default nccl linking
        end

        set_languages("cxx17")
        add_files("../src/infiniccl/cuda/*.cu")
    end
target_end()
