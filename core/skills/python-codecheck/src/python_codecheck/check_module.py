"""模块检查模式"""
from pathlib import Path
from typing import List

from .checker import check_files
from .utils import is_excluded_path


def check_module(paths: List[str]) -> bool:
    """
    模块检查模式
    
    Args:
        paths: 用户指定的路径列表
    
    Returns:
        是否全部通过
    """
    print("=== 模块检查模式 ===")
    
    if not paths:
        print("❌ 未指定检查路径")
        print("\n用法:")
        print("  python -m python_codecheck.main src/agent_sdk")
        print("  CHECK_PATHS=src/agent_sdk python -m python_codecheck.main")
        return False
    
    # 收集Python文件
    py_files = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if not path.exists():
            print(f"⚠️  路径不存在: {path}")
            continue
        
        if path.is_file():
            if path.suffix == ".py" and not is_excluded_path(path):
                py_files.append(path)
        else:
            # 目录：收集所有Python文件
            for py_file in path.rglob("*.py"):
                if not is_excluded_path(py_file):
                    py_files.append(py_file)
    
    return check_files(py_files)
