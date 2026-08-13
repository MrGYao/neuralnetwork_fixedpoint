"""公共检查器函数"""
import subprocess
from pathlib import Path
from typing import List, Tuple

from .find_project_root import find_project_root_for_file
from .utils import is_excluded_path


def run_mypy(files: List[Path], project_root: Path) -> Tuple[bool, str]:
    """
    运行mypy类型检查
    
    Args:
        files: 文件列表
        project_root: 项目根目录
    
    Returns:
        (是否成功, 错误信息)
    """
    print("  [1/3] mypy类型检查...")
    
    file_args = [str(f) for f in files if f.exists()]
    if not file_args:
        return True, ""
    
    args = ["uv", "run", "mypy"] + file_args
    
    result = subprocess.run(
        args,
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("    ✅ mypy检查通过")
        return True, ""
    
    print("    ❌ mypy检查失败")
    return False, result.stdout + result.stderr


def run_ruff_format(files: List[Path], project_root: Path) -> bool:
    """
    运行ruff格式化
    
    Args:
        files: 文件列表
        project_root: 项目根目录
    
    Returns:
        是否成功
    """
    print("  [2/3] ruff格式化...")
    
    file_args = [str(f) for f in files if f.exists()]
    if not file_args:
        return True
    
    args = ["ruff", "format"] + file_args
    
    result = subprocess.run(
        args,
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        changed = result.stdout.strip()
        if changed:
            print("    ✅ 已格式化")
        else:
            print("    ✅ 无需格式化")
        return True
    
    print(f"    ⚠️  格式化警告: {result.stderr}")
    return True


def run_ruff_check(files: List[Path], project_root: Path) -> Tuple[bool, str]:
    """
    运行ruff检查并自动修复
    
    Args:
        files: 文件列表
        project_root: 项目根目录
    
    Returns:
        (是否成功, 错误信息)
    """
    print("  [3/3] ruff检查...")
    
    file_args = [str(f) for f in files if f.exists()]
    if not file_args:
        return True, ""
    
    # 先尝试自动修复
    args_fix = ["ruff", "check", "--fix"] + file_args
    subprocess.run(
        args_fix,
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    # 再检查是否还有问题
    args_check = ["ruff", "check"] + file_args
    result = subprocess.run(
        args_check,
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("    ✅ ruff检查通过")
        return True, ""
    
    print("    ❌ ruff检查失败（无法自动修复）")
    return False, result.stdout


def check_files(files: List[Path]) -> bool:
    """
    检查文件列表
    
    Args:
        files: 文件列表
    
    Returns:
        是否全部通过
    """
    from collections import defaultdict
    
    if not files:
        print("无Python文件需要检查")
        return True
    
    print(f"检查文件: {len(files)} 个")
    for f in files:
        print(f"  - {f}")
    print()
    
    # 按项目分组
    project_files = defaultdict(list)
    for file_path in files:
        project_root = find_project_root_for_file(file_path)
        project_files[project_root].append(file_path)
    
    all_passed = True
    
    for project_root, project_files_list in project_files.items():
        print(f"项目: {project_root.name}")
        
        # mypy
        success, error = run_mypy(project_files_list, project_root)
        if not success:
            print(error)
            all_passed = False
            continue
        
        # ruff format
        run_ruff_format(project_files_list, project_root)
        
        # ruff check
        success, error = run_ruff_check(project_files_list, project_root)
        if not success:
            print(error)
            all_passed = False
        
        print()
    
    return all_passed
