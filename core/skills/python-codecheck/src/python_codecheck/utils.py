"""工具函数"""
import subprocess
from pathlib import Path
from typing import List, Optional


def run_command(
    args: List[str],
    cwd: Optional[Path] = None,
    capture: bool = True
) -> subprocess.CompletedProcess:
    """
    执行命令
    
    Args:
        args: 命令参数列表
        cwd: 工作目录
        capture: 是否捕获输出
    
    Returns:
        CompletedProcess对象
    """
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=capture,
        text=True
    )


def is_git_repo() -> bool:
    """检查是否在Git仓库中"""
    try:
        result = run_command(["git", "rev-parse", "--is-inside-work-tree"])
        return result.returncode == 0
    except:
        return False


def get_git_root() -> Optional[Path]:
    """获取Git根目录"""
    try:
        result = run_command(["git", "rev-parse", "--show-toplevel"])
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except:
        pass
    return None


def has_uncommitted_changes() -> bool:
    """检查是否有未提交的修改"""
    result1 = run_command(["git", "diff", "--quiet"])
    result2 = run_command(["git", "diff", "--cached", "--quiet"])
    return result1.returncode != 0 or result2.returncode != 0


def is_excluded_path(path: Path) -> bool:
    """
    检查路径是否应排除
    
    Args:
        path: 文件路径
    
    Returns:
        是否应排除
    """
    exclude_patterns = [
        "__pycache__",
        ".venv",
        "node_modules",
        ".git",
        ".tox",
        ".eggs",
        "*.egg-info",
        "site-packages",
    ]
    
    path_str = str(path)
    for pattern in exclude_patterns:
        if pattern in path_str:
            return True
    
    return False
