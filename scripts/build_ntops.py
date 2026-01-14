import concurrent.futures
import importlib
import pathlib

from infiniop.ninetoothed.build import BUILD_DIRECTORY_PATH

CURRENT_FILE_PATH = pathlib.Path(__file__)

SRC_DIR_PATH = CURRENT_FILE_PATH.parent.parent / "src"


def _find_and_build_ops():
    ops_path = SRC_DIR_PATH / "infiniop" / "ops"

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        for op_dir in ops_path.iterdir():
            ninetoothed_path = op_dir / "ninetoothed"

            if not ninetoothed_path.is_dir():
                continue

            futures.append(executor.submit(_build, ninetoothed_path))

        concurrent.futures.as_completed(futures)


def _build(ninetoothed_path):
    module_path = ninetoothed_path / "build"
    relative_path = module_path.relative_to(SRC_DIR_PATH)
    import_name = ".".join(relative_path.parts)
    module = importlib.import_module(import_name)

    module.build()


if __name__ == "__main__":
    BUILD_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)

    _find_and_build_ops()
