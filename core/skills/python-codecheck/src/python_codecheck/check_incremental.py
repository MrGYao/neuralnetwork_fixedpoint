"""增量检查逻辑（Working Directory Diff）"""
from pathlib import Path
from typing import List

from .checker import check_files
from .utils import run_command, is_excluded_path


def get_changed_py_files() -> List[Path]:
    """
    获取未提交的修改文件
    
    Returns:
        修改的.py文件路径列表
    """
    result = run_command(["git", "diff", "--name-only", "--diff-filter=ACMR"])
    
    if result.returncode != 0:
        return []
    
    files = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        
        path = Path(line)
        if path.suffix == ".py" and not is_excluded_path(path):
            files.append(path)
    
    return files


def check_incremental() -> bool:
    """
    执行增量检查
    
    Returns:
        是否全部通过
    """
    print("=== 增量检查模式 ===")
    
    changed_files = get_changed_py_files()
    
    if not changed_files:
        print("无Python文件修改")
        return True
    
    print("修改来源: Working Directory Diff\n")
    return check_files(changed_files)


if __name__ == "__main__":
    success = check_incremental()
    exit(0 if success else 1)
