"""Python代码质量检查主入口"""
import sys
from pathlib import Path

from .args import parse_args, detect_mode
from .check_tools import check_and_install_tools
from .inject_config import inject_ruff_config
from .find_project_root import find_project_root
from .check_incremental import check_incremental
from .check_module import check_module
from .check_commit_diff import check_commit_diff
from .check_full import check_full
from .commit_fix import commit_fix
from .utils import has_uncommitted_changes, is_git_repo


def main(argv=None):
    """主入口"""
    # 解析参数
    args = parse_args(argv)
    
    # 检测git状态
    is_git = is_git_repo()
    has_uncommitted = has_uncommitted_changes() if is_git else False
    
    # 确定最终模式
    mode = detect_mode(args, has_uncommitted, is_git)
    
    # 打印banner
    print("╔════════════════════════════════════╗")
    print("║    Python代码质量检查   ║")
    print("╚════════════════════════════════════╝")
    print()
    
    # Step 1: 工具检测与安装
    print("Step 1/4: 工具检测与安装")
    status = check_and_install_tools()
    
    failed = [k for k, v in status.items() if v == "failed"]
    if failed:
        print(f"❌ 工具安装失败: {', '.join(failed)}")
        return 1
    print()
    
    # Step 2: 配置注入
    print("Step 2/4: 配置注入")
    project_root = find_project_root()
    inject_ruff_config(project_root)
    print()
    
    # Step 3: 模式确认
    print("Step 3/4: 检查模式确认")
    print(f"  模式: {mode}")
    print(f"  Git仓库: {'是' if is_git else '否'}")
    if is_git:
        print(f"  未提交修改: {'是' if has_uncommitted else '否'}")
    if args.paths:
        print(f"  指定路径: {', '.join(args.paths)}")
    if mode == "commit-diff":
        print(f"  对比范围: 最近{args.compare_n}次提交")
    print()
    
    # Step 4: 质量检查循环
    print(f"Step 4/4: 质量检查循环 (最多{args.max_iterations}次)")
    return quality_check_loop(mode, args)


def quality_check_loop(mode: str, args) -> int:
    """
    质量检查循环
    
    Args:
        mode: 检查模式
        args: 参数对象
    
    Returns:
        退出码
    """
    max_iterations = args.max_iterations
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"迭代 {iteration}/{max_iterations}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # 执行检查
        success = run_check(mode, args)
        
        # 检查是否通过
        if success:
            print("\n╔════════════════════════════════════╗")
            print("║      ✅ 质量检查通过               ║")
            print("╚════════════════════════════════════╝")
            
            # 提示用户审视修复
            if has_uncommitted_changes():
                print("\n💡 已自动修复，请审视后提交:")
                import subprocess
                subprocess.run(["git", "diff", "--stat"])
            return 0
        
        # 检查是否有修复内容
        if not has_uncommitted_changes():
            print("\n❌ 检查失败但无修复内容，需要手动处理")
            print("\n💡 建议:")
            print("  1. 查看上述错误信息")
            print("  2. 手动修复问题")
            print("  3. 重新运行质量检查")
            return 1
        
        # 自动提交修复
        if args.auto_commit:
            print("\n自动修复已提交，继续检查...")
            commit_fix(iteration)
        else:
            print("\n💡 已自动修复，请手动提交后继续:")
            import subprocess
            subprocess.run(["git", "diff", "--stat"])
            return 0
    
    # 达到最大迭代次数
    print("\n╔════════════════════════════════════╗")
    print(f"║   ❌ 达到最大迭代次数 ({max_iterations})  ║")
    print("╚════════════════════════════════════╝")
    print("\n💡 建议:")
    print("  1. 查看最后一次检查的错误信息")
    print("  2. 手动修复无法自动解决的问题")
    print("  3. 重新运行质量检查")
    print(f"\n最后提交: ", end="")
    import subprocess
    subprocess.run(["git", "log", "-1", "--oneline"])
    return 1


def run_check(mode: str, args) -> bool:
    """
    执行检查
    
    Args:
        mode: 检查模式
        args: 参数对象
    
    Returns:
        是否通过
    """
    if mode == "module":
        return check_module(args.paths)
    elif mode == "incremental":
        return check_incremental()
    elif mode == "commit-diff":
        return check_commit_diff(args.compare_n)
    else:  # full
        return check_full()


if __name__ == "__main__":
    sys.exit(main())
