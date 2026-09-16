-- Resolve and inspect the exact extension linked into infinicore_cpp_api.
-- MACA's release number alone does not determine the varlen C++ ABI.
function resolve(root)
    local override = os.getenv("FLASH_ATTN_2_CUDA_SO")
    if override and override ~= "" then
        assert(os.isfile(override), "FLASH_ATTN_2_CUDA_SO is not a file: " .. override)
        return override
    end
    if root and root ~= "" then
        local files = os.files(path.join(root, "flash_attn_2_cuda*.so"))
        assert(#files <= 1, "Multiple MetaX extensions found; set FLASH_ATTN_2_CUDA_SO")
        if #files == 1 then return files[1] end
    end
    local container = os.getenv("FLASH_ATTN_METAX_CUDA_SO_CONTAINER")
    if container and container ~= "" then
        assert(os.isfile(container), "MetaX container extension is not a file: " .. container)
        return container
    end
    local python = os.getenv("PYTHON") or "python"
    local file = os.iorunv(python, {"-c",
        "import importlib.util; s=importlib.util.find_spec('flash_attn_2_cuda'); print(s.origin if s else '')"}):trim()
    assert(os.isfile(file), "MetaX flash_attn_2_cuda not found; set FLASH_ATTN_2_CUDA_SO")
    return file
end

function detect_symbols(symbols)
    local parameters = {
        "at::Tensor&", "at::Tensor const&", "at::Tensor const&",
        "std::optional<at::Tensor>&", "at::Tensor const&", "at::Tensor const&",
        "std::optional<at::Tensor>&", "std::optional<at::Tensor const>&",
        "std::optional<at::Tensor>&", "std::optional<at::Tensor>&",
        "int", "int", "float", "float", "bool", "bool", "int", "int",
        "float", "bool", "std::optional<at::Generator>"
    }
    local base = "mha_varlen_fwd(" .. table.concat(parameters, ", ")
    local signatures = {
        [base .. ")"] = {extension = false, return_max_logit = false},
        [base .. ", std::optional<at::Tensor>&)"] = {extension = true, return_max_logit = false},
        [base .. ", std::optional<at::Tensor>&, bool)"] = {extension = true, return_max_logit = true}
    }
    local found = nil
    for line in symbols:gmatch("[^\r\n]+") do
        local signature = line:match("^%s*%x+%s+[TW]%s+(.+)$")
        if signature and signature:find("mha_varlen_fwd(", 1, true) == 1 then
            assert(signatures[signature], "Unsupported MetaX Flash Attention ABI: " .. signature)
            assert(not found, "Ambiguous MetaX mha_varlen_fwd overloads")
            found = signatures[signature]
        end
    end
    assert(found, "MetaX extension does not export a supported mha_varlen_fwd; check FLASH_ATTN_2_CUDA_SO")
    return found
end

function configure(target, root)
    local file = resolve(root)
    local abi = detect_symbols(os.iorunv("nm", {"-D", "-C", "--defined-only", file}))
    local defines = {}
    if abi.extension then table.insert(defines, "INFINICORE_METAX_VARLEN_EXT") end
    if abi.return_max_logit then table.insert(defines, "INFINICORE_METAX_VARLEN_RETURN_MAX_LOGIT") end
    for _, define in ipairs(defines) do
        target:add("defines", define)
        target:add("cxflags", "-D" .. define)
        target:add("cxxflags", "-D" .. define)
    end
    print(string.format("MetaX varlen ABI: extension=%s, return_max_logit=%s (%s)",
        tostring(abi.extension), tostring(abi.return_max_logit), file))
end
