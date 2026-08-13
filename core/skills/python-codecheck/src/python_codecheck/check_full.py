"""全量检查逻辑"""
import subprocess
import tomllib
from pathlib import Path
from typing import List

from .find_project_root import find_project_root


def get_workspace_members(project_root: Path) -> List[str]:
    """
    获取monorepo的workspace成员
    
    Args:
        project_root: 项目根目录
    
    Returns:
        workspace成员路径列表
    """
    pyproject = project_root / "pyproject.toml"
    
    if not pyproject.exists():
        return ["."]
    
    with open(pyproject, "rb") as f:
        config = tomllib.load(f)
    
    members = config.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    
    return members if members else ["."]


def run_mypy_full(target_path: Path) -> tuple:
    """
    对目录运行mypy检查
    
    Args:
        target_path: 目标目录
    
    Returns:
        (是否成功, 错误信息)
    """
    print("  [1/3] mypy类型检查...")
    
    # 查找src目录
    src_path = target_path / "src"
    check_path = src_path if src_path.exists() else target_path
    
    args = ["uv", "run", "mypy", str(check_path)]
    
    result = subprocess.run(
        args,
        cwd=target_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("    ✅ mypy检查通过")
        return True, ""
    
    print("    ❌ mypy检查失败")
    return False, result.stdout + result.stderr


def run_ruff_format_full(target_path: Path) -> bool:
    """
    对目录运行ruff格式化
    
    Args:
        target_path: 目标目录
    
    Returns:
        是否成功
    """
    print("  [2/3] ruff格式化...")
    
    args = ["ruff", "format", "."]
    
    result = subprocess.run(
        args,
        cwd=target_path,
        capture_output=True,
        text=True
    )
    
    print("    ✅ 格式化完成")
    return True


def run_ruff_check_full(target_path: Path) -> tuple:
    """
    对目录运行ruff检查
    
    Args:
        target_path: 目标目录
    
    Returns:
        (是否成功, 错误信息)
    """
    print("  [3/3] ruff检查...")
    
    # 先自动修复
    args_fix = ["ruff", "check", "--fix", "."]
    subprocess.run(
        args_fix,
        cwd=target_path,
        capture_output=True,
        text=True
    )
    
    # 再检查
    args_check = ["ruff", "check", "."]
    result = subprocess.run(
        args_check,
        cwd=target_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("    ✅ ruff检查通过")
        return True, ""
    
    print("    ❌ ruff检查失败")
    return False, result.stdout


def check_full() -> bool:
    """
    执行全量检查
    
    Returns:
        是否全部通过
    """
    print("=== 全量检查模式 ===")
    
    project_root = find_project_root()
    members = get_workspace_members(project_root)
    
    if len(members) > 1:
        print(f"检测到monorepo结构 ({len(members)}个成员)")
    
    print(f"检查范围: {', '.join(members)}")
    print()
    
    all_passed = True
    
    for member in members:
        member_path = project_root / member
        
        if not member_path.exists():
            print(f"⚠️  跳过不存在的成员: {member}")
            continue
        
        print(f"━━━ {member} ━━━")
        
        # mypy
        success, error = run_mypy_full(member_path)
        if not success:
            print(error)
            all_passed = False
            continue
        
        # ruff format
        run_ruff_format_full(member_path)
        
        # ruff check
        success, error = run_ruff_check_full(member_path)
        if not success:
            print(error)
            all_passed = False
        
        print()
    
    return all_passed


if __name__ == "__main__":
    success = check_full()
    exit(0 if success else 1)
