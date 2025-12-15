from pathlib import Path

class TestDiscoverer:
    def __init__(self, ops_dir_path=None):
        self.ops_dir = self._resolve_dir(ops_dir_path)

    def _resolve_dir(self, path):
        if path:
            p = Path(path)
            if p.exists(): return p
        # 默认回退逻辑：当前文件的上级目录下的 ops
        # 注意：这里假设 lib/discovery.py 被引用，需要根据实际项目结构调整
        # 建议在 run.py 传入明确路径，这里只做辅助查找
        fallback = Path(__file__).parent.parent / "ops" 
        return fallback if fallback.exists() else None

    def get_available_operators(self):
        """返回所有可用算子的名称列表"""
        if not self.ops_dir: return []
        files = self.scan()
        return sorted([f.stem for f in files])

    def scan(self, specific_ops=None):
        """扫描并返回符合条件的 Path 对象列表"""
        if not self.ops_dir or not self.ops_dir.exists():
            return []

        # 1. 找所有 .py
        files = list(self.ops_dir.glob("*.py"))
        
        # 2. 过滤掉非测试文件（通过内容检查）
        valid_files = []
        for f in files:
            if f.name.startswith("_") or f.name == "run.py":
                continue
            if self._is_operator_test(f):
                valid_files.append(f)

        # 3. 如果指定了特定算子，进行筛选
        if specific_ops:
            return [f for f in valid_files if f.stem in specific_ops]
        
        return valid_files

    def _is_operator_test(self, file_path):
        """检查文件内容是否包含算子测试特征"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return "infinicore" in content and (
                    "BaseOperatorTest" in content or "GenericTestRunner" in content
                )
        except:
            return False
