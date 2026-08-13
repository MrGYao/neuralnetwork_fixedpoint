"""参数解析（环境变量 > CLI > 默认值）"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CheckArgs:
    """检查参数"""
    mode: str  # module/incremental/full/commit-diff
    paths: List[str]  # 要检查的路径
    compare_n: int  # 对比最近N次提交
    max_iterations: int  # 最大迭代次数
    auto_commit: bool  # 自动提交修复
    

def parse_args(argv: Optional[List[str]] = None) -> CheckArgs:
    """
    解析参数（环境变量 > CLI > 默认值）
    
    Args:
        argv: 命令行参数（用于测试）
    
    Returns:
        CheckArgs对象
    """
    parser = argparse.ArgumentParser(
        prog="python-codecheck",
        description="Python代码质量检查工具（跨平台，支持模块/增量/全量模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查指定模块
  python -m python_codecheck.main src/agent_sdk
  
  # 全量检查
  python -m python_codecheck.main --mode full
  
  # 对比最近2次提交
  python -m python_codecheck.main --mode commit-diff --compare-commits 2
  
  # 环境变量方式（agent调用友好）
  MODE=module CHECK_PATHS=src/agent_sdk python -m python_codecheck.main
        """
    )
    
    parser.add_argument(
        "paths",
        nargs="*",
        help="要检查的路径（支持多个）"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["module", "incremental", "full", "commit-diff"],
        help="检查模式: module(指定模块), incremental(增量), full(全量), commit-diff(对比提交)"
    )
    
    parser.add_argument(
        "--compare-commits", "-c",
        type=int,
        default=1,
        help="对比最近N次提交（commit-diff模式，默认1）"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="修复循环最大迭代次数（默认6）"
    )
    
    parser.add_argument(
        "--no-auto-commit",
        action="store_true",
        help="禁用自动提交修复"
    )
    
    args = parser.parse_args(argv)
    
    # 环境变量覆盖（优先级最高）
    mode = os.environ.get("MODE", args.mode)
    paths = args.paths
    compare_n = args.compare_commits
    max_iterations = args.max_iterations
    auto_commit = not args.no_auto_commit
    
    # 环境变量: CHECK_PATHS
    if os.environ.get("CHECK_PATHS"):
        paths = [p.strip() for p in os.environ["CHECK_PATHS"].split(",") if p.strip()]
    
    # 环境变量: COMPARE_N
    if os.environ.get("COMPARE_N"):
        compare_n = int(os.environ["COMPARE_N"])
    
    # 环境变量: MAX_ITERATIONS
    if os.environ.get("MAX_ITERATIONS"):
        max_iterations = int(os.environ["MAX_ITERATIONS"])
    
    # 环境变量: AUTO_COMMIT
    if os.environ.get("AUTO_COMMIT", "true").lower() == "false":
        auto_commit = False
    
    # 推断模式：如果指定了paths但mode为None，设置为module
    if paths and mode is None:
        mode = "module"
    
    return CheckArgs(
        mode=mode or "",  # 后续detect_mode会填充默认值
        paths=paths,
        compare_n=compare_n,
        max_iterations=max_iterations,
        auto_commit=auto_commit
    )


def detect_mode(args: CheckArgs, has_uncommitted: bool, is_git: bool) -> str:
    """
    检测最终检查模式
    
    Args:
        args: 用户参数
        has_uncommitted: 是否有未提交修改
        is_git: 是否Git仓库
    
    Returns:
        最终模式: module/incremental/full/commit-diff
    """
    # 用户明确指定模式
    if args.mode:
        return args.mode
    
    # 非Git仓库 → 全量
    if not is_git:
        return "full"
    
    # 有未提交修改 → 增量
    if has_uncommitted:
        return "incremental"
    
    # 无未提交修改 → 全量（用户决策）
    return "full"


if __name__ == "__main__":
    args = parse_args()
    print(f"mode: {args.mode}")
    print(f"paths: {args.paths}")
    print(f"compare_n: {args.compare_n}")
    print(f"max_iterations: {args.max_iterations}")
    print(f"auto_commit: {args.auto_commit}")
