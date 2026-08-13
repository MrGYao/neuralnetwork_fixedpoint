"""工具检测与安装"""
import shutil
import subprocess
from typing import Dict


def check_tool_installed(tool_name: str) -> bool:
    """
    检查工具是否已安装
    
    Args:
        tool_name: 工具名称
    
    Returns:
        是否已安装
    """
    return shutil.which(tool_name) is not None


def get_tool_version(tool_name: str) -> str:
    """
    获取工具版本
    
    Args:
        tool_name: 工具名称
    
    Returns:
        版本字符串
    """
    try:
        result = subprocess.run(
            [tool_name, "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split()
                if len(parts) >= 2:
                    return parts[1]
        return "unknown"
    except:
        return "unknown"


def install_tool(tool_name: str) -> bool:
    """
    使用uv tool install安装工具
    
    Args:
        tool_name: 工具名称
    
    Returns:
        是否安装成功
    """
    try:
        print(f"  正在安装 {tool_name}...")
        result = subprocess.run(
            ["uv", "tool", "install", tool_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"  ❌ 安装失败: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  ❌ uv未安装，请先安装uv: https://docs.astral.sh/uv/")
        return False


def check_and_install_tools() -> Dict[str, str]:
    """
    检测并安装所有必需工具
    
    Returns:
        工具状态字典 {"mypy": "installed", "ruff": "newly_installed", ...}
    """
    print("=== 工具检测与安装 ===")
    
    tools = ["mypy", "ruff"]
    status = {}
    
    for tool in tools:
        if check_tool_installed(tool):
            version = get_tool_version(tool)
            path = shutil.which(tool)
            print(f"✅ {tool} 已安装: v{version} ({path})")
            status[tool] = "installed"
        else:
            print(f"⚠️  {tool} 未安装")
            if install_tool(tool):
                version = get_tool_version(tool)
                print(f"✅ {tool} 安装完成: v{version}")
                status[tool] = "newly_installed"
            else:
                status[tool] = "failed"
    
    print()
    return status
