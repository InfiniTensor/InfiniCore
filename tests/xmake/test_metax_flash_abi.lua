function main()
    local abi = import("metax_flash_abi", {rootdir = path.join(os.projectdir(), "xmake")})
    local common = "mha_varlen_fwd(at::Tensor&, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor>&, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor>&, std::optional<at::Tensor const>&, std::optional<at::Tensor>&, std::optional<at::Tensor>&, int, int, float, float, bool, bool, int, int, float, bool, std::optional<at::Generator>"
    local function symbol(suffix) return "0000000000123456 T " .. common .. suffix .. "\n" end
    local old = abi.detect_symbols(symbol(")"))
    assert(not old.extension and not old.return_max_logit)
    local ext = abi.detect_symbols(symbol(", std::optional<at::Tensor>&)"))
    assert(ext.extension and not ext.return_max_logit)
    local latest = abi.detect_symbols(symbol(", std::optional<at::Tensor>&, bool)"))
    assert(latest.extension and latest.return_max_logit)
    for _, bad in ipairs({"", symbol(", int)"), symbol(")") .. symbol(", std::optional<at::Tensor>&)")}) do
        local failed = false
        try {function() abi.detect_symbols(bad) end,
             catch {function() failed = true end}}
        assert(failed)
    end
    if os.getenv("FLASH_ATTN_2_CUDA_SO") then
        local file = abi.resolve()
        local actual = abi.detect_symbols(os.iorunv("nm", {"-D", "-C", "--defined-only", file}))
        print("installed extension: " .. file)
        print(actual)
    end
    print("MetaX ABI: three signatures accepted; missing/unknown/ambiguous rejected")
end
