local CUDA_ROOT = os.getenv("CUDA_ROOT") or os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
local CUDNN_ROOT = os.getenv("CUDNN_ROOT") or os.getenv("CUDNN_HOME") or os.getenv("CUDNN_PATH")
if CUDA_ROOT ~= nil then
    add_includedirs(CUDA_ROOT .. "/include")
end
if CUDNN_ROOT ~= nil then
    add_includedirs(CUDNN_ROOT .. "/include")
end

if is_plat("windows") then
    set_runtimes("MD")
end

-- 静态库目标
target("infinicub")
    set_kind("static")
    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cublas", "cudnn")
    add_cugencodes("native")
  
    add_defines("ENABLE_CUDA_API")

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
        add_cxxflags("-fPIC")
    end

    set_languages("cxx17")
    add_includedirs("include")

    add_files("src/*.cu") 
target_end()
