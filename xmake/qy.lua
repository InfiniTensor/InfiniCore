local TORCH_DIR = os.getenv("TORCH_DIR")

if not TORCH_DIR then
    raise("TORCH_DIR is not set! please export it first")
end

print("TORCH_DIR =", TORCH_DIR)

if TORCH_DIR and os.isdir(TORCH_DIR) then
    local TORCH_INCLUDE = TORCH_DIR .. "/include"
    local TORCH_LIB = TORCH_DIR .. "/lib"
    
    print("✅ 自动找到 PyTorch 路径: " .. TORCH_DIR)
    print("✅ PyTorch 头文件: " .. TORCH_INCLUDE)
    print("✅ PyTorch 库路径: " .. TORCH_LIB)

    -- 添加 PyTorch 头文件
    add_includedirs(TORCH_INCLUDE)
    add_includedirs(TORCH_INCLUDE .. "/torch/csrc/api/include")

    -- 添加 PyTorch 库路径
    add_linkdirs(TORCH_LIB)

    -- 链接 PyTorch 核心库（解决 undefined symbol）
    add_links("torch", "torch_cpu", "torch_cuda", "c10", "c10_cuda", "torch_python")
else
    print("⚠️  未检测到 PyTorch，将跳过 PyTorch 依赖")
end


local CUDNN_ROOT = os.getenv("CUDNN_ROOT") or os.getenv("CUDNN_HOME") or os.getenv("CUDNN_PATH")
if CUDNN_ROOT ~= nil then
    add_includedirs(CUDNN_ROOT .. "/include")
end

add_includedirs("/usr/local/denglin/sdk/include", "../include")
add_linkdirs("/usr/local/denglin/sdk/lib")
add_links("curt", "cublas", "cudnn")
set_languages("cxx17")
add_cxxflags("-std=c++17")  -- 显式设置 C++17
add_cuflags("--std=c++17",{force = true})  -- 确保 CUDA 编译器也使用 C++17
rule("ignore.o")
    set_extensions(".o")  -- 防止 xmake 默认处理
    on_build_files(function () end)

rule("qy.cuda")
    set_extensions(".cu")

    -- 缓存所有 .o 文件路径
    local qy_objfiles = {}

    on_load(function (target)
        target:add("includedirs", "/usr/local/denglin/sdk/include")
        
        -- 把 PyTorch 头文件也加入自定义规则
        if TORCH_DIR then
            target:add("includedirs", TORCH_DIR .. "/include")
            target:add("includedirs", TORCH_DIR .. "/include/torch/csrc/api/include")
        end
    end)

    after_load(function (target)
        -- 过滤 cudadevrt/cudart_static
        local links = target:get("syslinks") or {}
        local filtered = {}
        for _, link in ipairs(links) do
            if link ~= "cudadevrt" and link ~= "cudart_static" then
                table.insert(filtered, link)
            end
        end
        target:set("syslinks", filtered)
    end)

    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        import("core.project.project")
        import("core.project.config")
        import("core.base.option")

        local dlcc = "/usr/local/denglin/sdk/bin/dlcc"
        local sdk_path = "/usr/local/denglin/sdk"
        local arch = "dlgput64"

        local relpath = path.relative(sourcefile, project.directory())
        local objfile = path.join(config.buildir(), ".objs", target:name(), "rules", "qy.cuda", relpath .. ".o")

        -- 🟢 强制注册 .o 文件给 target
        target:add("objectfiles", objfile)
        target:set("buildadd", true)
        local argv = {
            "-c", sourcefile,
            "-o", objfile,
            "--cuda-path=" .. sdk_path,
            "--cuda-gpu-arch=" .. arch,
            "-std=c++17", "-O2", "-fPIC"
        }

        for _, incdir in ipairs(target:get("includedirs") or {}) do
            table.insert(argv, "-I" .. incdir)
        end
        for _, def in ipairs(target:get("defines") or {}) do
            table.insert(argv, "-D" .. def)
        end

        batchcmds:mkdir(path.directory(objfile))
        batchcmds:show_progress(opt.progress, "${color.build.object}compiling.dlcu %s", relpath)
        batchcmds:vrunv(dlcc, argv)
    end)
target("infiniop-qy")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    add_rules("qy.cuda", {override = true})

    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cuflags("-Xcompiler=/W3", "-Xcompiler=/WX")
        add_cxxflags("/FS")
        if CUDNN_ROOT ~= nil then
            add_linkdirs(CUDNN_ROOT .. "\\lib\\x64")
        end
    else
        add_cuflags("-Xcompiler=-Wall", "-Xcompiler=-Werror")
        add_cuflags("-Xcompiler=-fPIC")
        add_cuflags("--extended-lambda")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")
        add_cuflags("--expt-relaxed-constexpr")
        if CUDNN_ROOT ~= nil then
            add_linkdirs(CUDNN_ROOT .. "/lib")
        end
    end

    add_cuflags("-Xcompiler=-Wno-error=deprecated-declarations")

    set_languages("cxx17")
    add_files("../src/infiniop/devices/nvidia/*.cu", "../src/infiniop/ops/*/nvidia/*.cu", "../src/infiniop/ops/*/*/nvidia/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", "../build/ninetoothed/*.cpp")
    end
target_end()

target("infinirt-qy")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)
    add_rules("qy.cuda", {override = true})
    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cxxflags("/FS")
    else
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")
    end

    set_languages("cxx17")
    add_files("../src/infinirt/cuda/*.cu")
target_end()

target("infiniccl-qy")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)
    if has_config("ccl") then
        add_rules("qy.cuda", {override = true})
        if not is_plat("windows") then
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
            add_cxflags("-fPIC")
            add_cxxflags("-fPIC")

            local nccl_root = os.getenv("NCCL_ROOT")
            if nccl_root then
                add_includedirs(nccl_root .. "/include")
                add_links(nccl_root .. "/lib/libnccl.so")
            else
                add_links("nccl") -- Fall back to default nccl linking
            end

            add_files("../src/infiniccl/cuda/*.cu")
        else
            print("[Warning] NCCL is not supported on Windows")
        end
    end
    set_languages("cxx17")

target_end()
