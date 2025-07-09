package("infinicub")
    set_description("Build infinicub library.")

    add_versions("1.0.0", "commit-hash-or-sha256-for-v1.0.0")
    add_versions("1.0.1", "commit-hash-or-sha256-for-v1.0.1")
    add_versions("1.0.2", "commit-hash-or-sha256-for-v1.0.2")
    add_versions("1.0.3", "commit-hash-or-sha256-for-v1.0.3")
    add_versions("1.0.4", "commit-hash-or-sha256-for-v1.0.4")

    set_sourcedir(path.join(os.scriptdir(), "../src/infinicub"))
  
    local dir = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "USERPROFILE") .. "/.infini")
    set_installdir(path.join(dir,"packages/infinicub/", get_config("plat"), get_config("arch"), get_config("mode")))

    add_configs("shared", {default = false, type = "boolean", readonly = true})
    on_install(function (package)
        local configs = {}
        import("package.tools.xmake").install(package, configs)
    end)
package_end()
