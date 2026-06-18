import os
import json
import logging
import multiprocessing
import shutil
import subprocess
import argparse
import fnmatch
from pathlib import Path
from safetensors import safe_open
import torch
import h5py
import numpy as np
from transformers import AutoModel
import gguf

from .infiniop_test import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides

parser = argparse.ArgumentParser( description="自动提取模型参数、生成 GGUF 测例并运行 infiniop-test" )
parser.add_argument( "--config", default="config-hub.json", help="模型列表配置文件，指定patterns" )
parser.add_argument( "--model-path", default=None, help="模型存储根目录：若指定，则扫描该目录下的模型文件夹" )
parser.add_argument( "--output", default="gguf_output", help="输出目录" )
parser.add_argument( "--warmup", type=int, default=20, help="预热次数" )
parser.add_argument( "--run", type=int, default=1000, help="测试次数" )
parser.add_argument( "--overwrite", action="store_true", help="覆盖已有输出" )
args = parser.parse_args()

# HuggingFace下载可能需启用代理
# os.environ['HTTP_PROXY'] = 'http://XXX.XXX.XXX.XXX:XXX'
# os.environ['HTTPS_PROXY'] = 'http://XXX.XXX.XXX.XXX:XXX'

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_EXTS = [ '*.bin', '*.safetensors', '*.pt', '*.pth', '*.ckpt*', '*.h5', '*.onnx', ]

# 筛选模型文件
def is_valid_candidate(path: Path) -> bool:
    if not any(fnmatch.fnmatch(path.name, pat) for pat in CANDIDATE_EXTS):
        return False
    size = path.stat().st_size
    if size < 1_024 or size > 50 * 1024**2:
        return False
    try:
        suf = path.suffix.lower()
        if suf in ('.pt', '.pth', '.bin'):
            try:
                _ = AutoModel.from_pretrained(path, local_files_only=True)
            except Exception:
                _ = torch.load(path, map_location='cpu')
        elif suf == '.safetensors':
            with safe_open(path, framework="pt", device="cpu") as f:
                _ = list(f.keys())
        elif suf.startswith('.ckpt'):
            pass
        elif suf in ('.h5',):
            with h5py.File(path, 'r'): pass
        elif suf == '.onnx':
            with open(path, 'rb') as f:
                hdr = f.read(4)
                if b'\n\x08\x01\x12' not in hdr:
                    raise ValueError("Invalid ONNX header")
        else:
            return False
    except Exception:
        return False
    return True

if args.model_path:
    root = Path(args.model_path)
    CUSTOM_MODELS = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if is_valid_candidate(f):
                CUSTOM_MODELS.append(sub)
                break
    print(f"Find {len(CUSTOM_MODELS)} model path：{[m.name for m in CUSTOM_MODELS]}")
    CONFIG = None
else:
    CONFIG = json.load(open(str(PROJECT_ROOT / "test" / "infiniop-test" / "test_generate" / args.config)))
    CUSTOM_MODELS = None

def compute_strides(arr: np.ndarray):
    return [s // arr.dtype.itemsize for s in arr.strides]

def extract_model(src, out_dir: Path, overwrite: bool):
    if isinstance(src, Path):
        model_name = src.name
        loader_kwargs = {'pretrained_model_name_or_path': str(src)}
    else:
        model_name = src
        loader_kwargs = {'pretrained_model_name_or_path': model_name}

    safe_name = model_name.replace('/', '_')
    model_dir = out_dir / safe_name
    if model_dir.exists() and not overwrite:
        print(f"Skip {model_name}")
        return
    if model_dir.exists():
        shutil.rmtree(model_dir)
    (model_dir / "logs").mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(str(model_dir / "logs" / f"{safe_name}.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info(f"Loading {model_name}")
    model = AutoModel.from_pretrained(**loader_kwargs, torch_dtype=torch.float16, device_map="cpu")

    # 匹配patterns
    if CONFIG and not args.model_path:
        patterns = CONFIG.get(src, [])
    else:
        patterns = ['*.weight']

    raw_matches = []
    for name, param in model.named_parameters():
        for pat in patterns:
            if fnmatch.fnmatch(name, pat):
                raw_matches.append((name, param))
    # 仅保留二维矩阵，跳过 1D/3D 等
    matched_params = []
    for name, param in raw_matches:
        arr = param.detach().cpu().numpy()
        if arr.ndim != 2:
            logger.warning(f"Skip non-2D parameters {name} (shape={arr.shape})")
            continue
        matched_params.append((name, param))

    if not matched_params:
        logger.error(f"{model_name} non-2D parameters available for GEMM")
        return

    logger.info(f"[{model_name}] Match {len(matched_params)} GEMM： {[n for n,_ in matched_params]}")

    # 生成 GGUF
    gguf_path = model_dir / "gemm.gguf"
    writer = InfiniopTestWriter(str(gguf_path))

    for pname, pval in matched_params:
        W = pval.detach().cpu().numpy().astype(np.float32)
        out_dim, in_dim = W.shape
        a = np.eye(in_dim, dtype=np.float32)[0:1, :]
        b = W.T
        c = np.zeros((1, out_dim), dtype=np.float32)
        alpha, beta = 1.0, 1.0

        class ModelGemmTest(InfiniopTestCase):
            def __init__(self, a, b, c, alpha, beta):
                super().__init__("gemm")
                self.a, self.b, self.c = a, b, c
                self.alpha, self.beta = alpha, beta

            def write_test(self, tw):
                super().write_test(tw)
                tw.add_array(tw.gguf_key("a.strides"), gguf_strides(*compute_strides(self.a)))
                tw.add_array(tw.gguf_key("b.strides"), gguf_strides(*compute_strides(self.b)))
                tw.add_array(tw.gguf_key("c.strides"), gguf_strides(*compute_strides(self.c)))
                tw.add_float32(tw.gguf_key("alpha"), self.alpha)
                tw.add_float32(tw.gguf_key("beta"), self.beta)
                tw.add_tensor(tw.gguf_key("a"), self.a, raw_dtype=np_dtype_to_ggml(self.a.dtype))
                tw.add_tensor(tw.gguf_key("b"), self.b, raw_dtype=np_dtype_to_ggml(self.b.dtype))
                tw.add_tensor(tw.gguf_key("c"), self.c, raw_dtype=np_dtype_to_ggml(self.c.dtype))
                ans = self.alpha * (self.a.astype(np.float64) @ self.b.astype(np.float64)) + \
                      self.beta * self.c.astype(np.float64)
                tw.add_tensor(tw.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64)

        writer.add_test(ModelGemmTest(a, b, c, alpha, beta))
    writer.save()
    logger.info(f"Generated {gguf_path}")


def build_infiniop_test():
    # 一次性编译整个项目（包括所有库和 infiniop-test）
    subprocess.run([
        "xmake", "-a"
    ], check=True, cwd=str(PROJECT_ROOT))


def run_test(gguf_file: Path, warmup: int, runs: int):
    build_dir = PROJECT_ROOT / "build" / "linux" / "x86_64" / "release"
    exe = build_dir / "infiniop-test"
    env = os.environ.copy()
    env["PATH"] = str(build_dir) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(build_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    print(f"Run the test {gguf_file}")
    subprocess.run([
        str(exe), str(gguf_file), "--cpu",
        "--warmup", str(warmup), "--run", str(runs)
    ], check=True, env=env)


if __name__ == "__main__":
    out = Path(args.output)
    out.mkdir(exist_ok=True)

    if args.model_path is not None:
        tasks = [(p, out, args.overwrite) for p in CUSTOM_MODELS]
    else:
        cfg_path = Path(__file__).resolve().parent / args.config
        models = json.load(open(cfg_path))
        tasks = [(m, out, args.overwrite) for m in models]

    with multiprocessing.Pool() as pool:
        pool.starmap(extract_model, tasks)

    build_infiniop_test()
    for src, _, _ in tasks:
        name = src.name if isinstance(src, Path) else src.replace('/', '_')
        gguf_file = out / name / "gemm.gguf"
        run_test(gguf_file, args.warmup, args.run)