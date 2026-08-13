"""提交修复"""
import subprocess
from pathlib import Path
from datetime import datetime
from .utils import has_uncommitted_changes


def commit_changes(message: str) -> bool:
    """
    提交所有修改
    
    Args:
        message: 提交消息
    
    Returns:
        是否有提交
    """
    if not has_uncommitted_changes():
        return False
    
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    return True


def commit_code_changes() -> bool:
    """
    提交编码成果
    
    Returns:
        是否有提交
    """
    print("=== 提交编码成果 ===")
    
    if not has_uncommitted_changes():
        print("无未提交的修改")
        return False
    
    result = subprocess.run(
        ["git", "diff", "--stat"],
        capture_output=True,
        text=True
    )
    
    message = f"""feat: 编码完成


修改统计:
{result.stdout.strip()}

提交时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    if commit_changes(message):
        print("✅ 已保存编码成果")
        return True
    
    return False


def commit_fix(iteration: int) -> bool:
    """
    提交自动修复
    
    Args:
        iteration: 迭代次数
    
    Returns:
        是否有提交
    """
    if not has_uncommitted_changes():
        print("无修改需要提交")
        return False
    
    result = subprocess.run(
        ["git", "diff", "--stat"],
        capture_output=True,
        text=True
    )
    
    message = f"""fix: 自动修复代码质量问题 (迭代{iteration})


执行工具:
- ruff format (代码格式化)
- ruff check --fix (lint自动修复)

修改统计:
{result.stdout.strip()}"""
    
    if commit_changes(message):
        print(f"✅ 提交修复 (迭代{iteration})")
        return True
    
    return False


if __name__ == "__main__":
    import sys
    
    iteration = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    if commit_fix(iteration):
        print("提交成功")
    else:
        print("无修改")
