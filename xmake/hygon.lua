local dtk_root = os.getenv("DTK_ROOT")
toolchain("hygon.toolchain")
    set_toolset("cc"  , "clang"  )
    set_toolset("cxx" , "clang++")
    -- 使用DTK中的CUDA编译器
    local nvcc_path = path.join(dtk_root, "cuda", "bin", "nvcc")
    if os.isfile(nvcc_path) then
        set_toolset("cu"  , nvcc_path)
        set_toolset("culd", nvcc_path)
    else
        set_toolset("cu"  , "nvcc")
        set_toolset("culd", "nvcc")
    end
    set_toolset("cu-ccbin", "$(env CXX)", "$(env CC)")
toolchain_end()

rule("hygon.env")
    -- Fix the deprecated warning by using add_orders
    add_orders("cuda.env", "hygon.env")
    after_load(function (target)
        -- This logic to remove CUDA-specific libs is correct and can remain
        local old = target:get("syslinks") or {}
        local new = {}
        for _, link in ipairs(old) do
            if link ~= "cudadevrt" and link ~= "cudnn" then
                table.insert(new, link)
            end
        end
        if #old > #new then
            target:set("syslinks", new)
            print("CUDA specific libraries removed for Hygon DCU. New syslinks: {" .. table.concat(new, ", ") .. "}")
        end
    end)
rule_end()

target("infiniop-hygon")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("hygon.toolchain")
    add_rules("hygon.env")
    set_values("cuda.rdc", false)

    -- 海光DCU使用DTK中的CUDA库
    add_links("cudart", "cublas", "curand", "cublasLt", "cudnn")
    
    -- 添加DTK路径支持
    local dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
    if os.isdir(dtk_root) then
        add_includedirs(path.join(dtk_root, "include"))
        add_includedirs(path.join(dtk_root, "cuda", "include"))
        add_linkdirs(path.join(dtk_root, "lib"))
        add_linkdirs(path.join(dtk_root, "cuda", "lib64"))
    end

    set_warnings("all", "error")
    add_cuflags("-Wno-error=unused-private-field")
    add_cuflags("-Wno-return-type", {force = true})  -- 抑制return语句警告
    add_cuflags("-fPIC", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")

    -- 添加海光DCU特定的编译标志
    add_cuflags("-arch=gfx906", "-arch=gfx926", "-arch=gfx928", "-arch=gfx936")
    
    -- 复用NVIDIA的CUDA实现，通过HIP兼容层
    -- 只编译海光DCU支持的7个算子：rope, gemm, causal_softmax, random_sample, rearrange, rms_norm, swiglu
    add_files("../src/infiniop/devices/nvidia/*.cu")
    add_files("../src/infiniop/ops/rope/nvidia/*.cu")
    add_files("../src/infiniop/ops/gemm/nvidia/*.cu")
    add_files("../src/infiniop/ops/causal_softmax/nvidia/*.cu")
    add_files("../src/infiniop/ops/random_sample/nvidia/*.cu")
    add_files("../src/infiniop/ops/rearrange/nvidia/*.cu")
    add_files("../src/infiniop/ops/rms_norm/nvidia/*.cu")
    add_files("../src/infiniop/ops/swiglu/nvidia/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", {cxflags = {"-Wno-return-type"}})
    end
target_end()

target("infinirt-hygon")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("hygon.toolchain")
    add_rules("hygon.env")
    set_values("cuda.rdc", false)

    add_links("cudart", "curand")
    
    -- 添加DTK路径支持
    local dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
    if os.isdir(dtk_root) then
        add_includedirs(path.join(dtk_root, "include"))
        add_includedirs(path.join(dtk_root, "cuda", "include"))
        add_linkdirs(path.join(dtk_root, "lib"))
        add_linkdirs(path.join(dtk_root, "cuda", "lib64"))
    end

    set_warnings("all", "error")
    add_cuflags("-Wno-return-type", {force = true})  -- 抑制return语句警告
    add_cuflags("-fPIC", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")

    -- 添加海光DCU特定的编译标志
    add_cuflags("-arch=gfx906", "-arch=gfx926", "-arch=gfx928", "-arch=gfx936")
    
    add_files("../src/infinirt/cuda/*.cu")
target_end()

target("infiniccl-hygon")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)

    if has_config("ccl") then
        set_toolchains("hygon.toolchain")
        add_rules("hygon.env")
        set_values("cuda.rdc", false)

        add_links("cudart", "curand")
        
        -- 添加DTK路径支持
        local dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
        if os.isdir(dtk_root) then
            add_includedirs(path.join(dtk_root, "include"))
            add_includedirs(path.join(dtk_root, "cuda", "include"))
            add_linkdirs(path.join(dtk_root, "lib"))
            add_linkdirs(path.join(dtk_root, "cuda", "lib64"))
        end

        set_warnings("all", "error")
        add_cuflags("-Wno-return-type", {force = true})  -- 抑制return语句警告
        add_cuflags("-fPIC", "-std=c++17", {force = true})
        add_culdflags("-fPIC")
        add_cxflags("-fPIC")

        -- 添加海光DCU特定的编译标志
        add_cuflags("-arch=gfx906", "-arch=gfx926", "-arch=gfx928", "-arch=gfx936")

        -- 使用NCCL (NVIDIA Collective Communications Library)
        add_links("nccl")

        add_files("../src/infiniccl/cuda/*.cu")
    end
target_end()
