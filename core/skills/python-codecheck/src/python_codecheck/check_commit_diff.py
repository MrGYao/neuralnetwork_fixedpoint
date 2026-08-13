"""Commit对比检查模式"""
from pathlib import Path
from typing import List

from .checker import check_files
from .utils import run_command, is_excluded_path


def get_commit_diff_files(compare_n: int = 1) -> List[Path]:
    """
    获取最近N次提交的修改文件
    
    Args:
        compare_n: 对比最近N次提交
    
    Returns:
        修改的.py文件路径列表
    """
    # git diff HEAD~N..HEAD --name-only
    result = run_command([
        "git", "diff",
        f"HEAD~{compare_n}..HEAD",
        "--name-only",
        "--diff-filter=ACMR"
    ])
    
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


def check_commit_diff(compare_n: int = 1) -> bool:
    """
    Commit对比检查模式
    
    Args:
        compare_n: 对比最近N次提交
    
    Returns:
        是否全部通过
    """
    print("=== Commit对比检查模式 ===")
    
    # 检查git历史是否足够
    result = run_command(["git", "rev-list", "--count", "HEAD"])
    if result.returncode != 0:
        print("❌ 无法获取git提交历史")
        return False
    
    commit_count = int(result.stdout.strip())
    if commit_count < compare_n:
        print(f"⚠️  提交历史不足{compare_n}次，当前{commit_count}次提交")
        compare_n = commit_count
    
    print(f"对比范围: HEAD~{compare_n}..HEAD")
    
    # 获取修改文件
    diff_files = get_commit_diff_files(compare_n)
    
    if not diff_files:
        print("无Python文件修改")
        return True
    
    # 显示commit信息
    result = run_command(["git", "log", f"--oneline", f"-{compare_n}"])
    if result.returncode == 0:
        print("\n最近提交:")
        for line in result.stdout.strip().split("\n")[:compare_n]:
            print(f"  {line}")
        print()
    
    return check_files(diff_files)
