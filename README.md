# InfiniCore

InfiniCore is the version manifest for the InfiniTensor core stack. It pins three independently developed components as Git submodules:

- [InfiniRT](https://github.com/InfiniTensor/InfiniRT) provides device and runtime services.
- [InfiniOps](https://github.com/InfiniTensor/InfiniOps) provides computational operators.
- [InfiniCCL](https://github.com/InfiniTensor/InfiniCCL) provides collective communication.

## Clone and update

Clone the repository with its pinned submodules:

```shell
git clone --recurse-submodules https://github.com/InfiniTensor/InfiniCore.git
```

Initialize or update the submodules in an existing checkout:

```shell
git submodule sync --recursive
git submodule update --init --recursive
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and validation
guidelines.

## License

InfiniCore is licensed under the MIT License. See [LICENSE](LICENSE).
