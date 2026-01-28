toolchain("ali.toolchain")
    set_toolset("cc"  , "clang"  )
    set_toolset("cxx" , "clang++")
    set_toolset("cu"  , "/usr/local/PPU_SDK/CUDA_SDK/bin/clang++")
    set_toolset("culd", "clang++")
    set_toolset("cu-ccbin", "$(env CXX)", "$(env CC)")
    
    on_load(function (toolchain)
        toolchain:add("cxflags", "-std=c++17")
        toolchain:add("cxxflags", "-std=c++17")

        local cuda_path = "/usr/local/cuda"
        toolchain:add("cuflags", "--cuda-path=" .. cuda_path)
        toolchain:add("cuflags", "-xc++")
        toolchain:add("cuflags", "-std=c++17")
        toolchain:add("cuflags", "-Qunused-arguments")
        toolchain:add("cuflags", "-Wno-c++17-extensions")
        toolchain:add("cuflags", "-Wno-error=c++17-extensions")
        toolchain:add("cuflags", "-Wno-ignored-attributes")
        toolchain:add("cuflags", "-Wno-unknown-attributes")

        toolchain:add("cuflags", "-D__CUDACC__")
        toolchain:add("cuflags", "-D__NVCC__")
        toolchain:add("cuflags", "-D__CUDA_NO_HALF_OPERATORS__")
        toolchain:add("cuflags", "-D__CUDA_NO_HALF_CONVERSIONS__")
        toolchain:add("cuDA_NO_BFLOAT16_CONVERSIONS__")

        toolchain:add("cuflags", "--cuda-gpu-arch=sm_80")

        toolchain:add("includedirs", cuda_path .. "/include")
        
        toolchain:add("cxxflags", "-Wno-c++17-extensions")
        toolchain:add("cxxflags", "-Wno-error=c++17-extensions")
        toolchain:add("cxxflags", "-Wno-ignored-attributes")
        toolchain:add("cxxflags", "-Wno-unknown-attributes")
    end)
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
    -- add_cuflags("-fPIC", "-xc++", {force = true})
    add_cuflags("-fPIC", "-xc++", "-std=c++17", {force = true})
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
    -- add_cuflags("-fPIC", "-xc++", {force = true})
    add_cuflags("-fPIC", "-xc++", "-std=c++17", {force = true})
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
        -- add_cuflags("-fPIC", "-xc++", {force = true})
        add_cuflags("-fPIC", "-xc++", "-std=c++17", {force = true})
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
