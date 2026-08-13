"""配置注入"""
import tomllib
from pathlib import Path
from typing import Dict, Any, Optional, List

from .templates import get_default_ruff_config, get_default_mypy_config


def load_toml(file_path: Path) -> Dict[str, Any]:
    """
    加载TOML文件
    
    Args:
        file_path: TOML文件路径
    
    Returns:
        配置字典
    """
    if not file_path.exists():
        return {}
    
    with open(file_path, "rb") as f:
        return tomllib.load(f)


def save_toml(data: Dict[str, Any], file_path: Path):
    """
    保存TOML文件
    
    需要tomli-w库: pip install tomli-w
    
    Args:
        data: 配置字典
        file_path: 目标文件路径
    """
    try:
        import tomli_w
    except ImportError:
        print("⚠️  tomli-w未安装，正在安装...")
        import subprocess
        import shutil
        
        if shutil.which("uv"):
            subprocess.run(["uv", "pip", "install", "tomli-w"], check=True)
        else:
            subprocess.run(["pip", "install", "tomli-w"], check=True)
        
        import tomli_w
    
    with open(file_path, "wb") as f:
        tomli_w.dump(data, f)


def has_ruff_config(config: Dict[str, Any]) -> bool:
    """
    检查配置中是否已有ruff配置
    
    Args:
        config: 配置字典
    
    Returns:
        是否已有ruff配置
    """
    return "tool" in config and "ruff" in config.get("tool", {})


def has_mypy_config(config: Dict[str, Any]) -> bool:
    """
    检查配置中是否已有mypy配置
    
    Args:
        config: 配置字典
    
    Returns:
        是否已有mypy配置
    """
    return "tool" in config and "mypy" in config.get("tool", {})


def get_package_name(config: Dict[str, Any]) -> Optional[str]:
    """
    从pyproject.toml获取包名
    
    Args:
        config: pyproject.toml配置
    
    Returns:
        包名或None
    """
    return config.get("project", {}).get("name")


def inject_ruff_config(
    project_root: Path,
    line_length: int = 100,
    python_version: str = "py310"
) -> bool:
    """
    注入ruff配置到项目
    
    策略：
    1. 如果存在pyproject.toml且无[tool.ruff]，追加配置
    2. 如果存在pyproject.toml且有[tool.ruff]，跳过
    3. 如果不存在pyproject.toml，创建.ruff.toml
    
    Args:
        project_root: 项目根目录
        line_length: 行长度限制
        python_version: Python版本
    
    Returns:
        是否成功
    """
    print("=== ruff配置注入 ===")
    
    pyproject_path = project_root / "pyproject.toml"
    ruff_toml_path = project_root / ".ruff.toml"
    
    if pyproject_path.exists():
        print(f"检测到pyproject.toml: {pyproject_path}")
        config = load_toml(pyproject_path)
        
        if has_ruff_config(config):
            print("✅ 已有ruff配置，跳过注入")
            return True
        
        print("注入ruff配置到pyproject.toml...")
        
        config.setdefault("tool", {})
        
        package_name = get_package_name(config)
        known_first_party = [package_name] if package_name else []
        
        config["tool"]["ruff"] = get_default_ruff_config(
            line_length=line_length,
            python_version=python_version,
            known_first_party=known_first_party
        )
        
        backup_path = pyproject_path.with_suffix(".toml.backup")
        if pyproject_path.exists():
            import shutil
            shutil.copy(pyproject_path, backup_path)
            print(f"  备份: {backup_path}")
        
        save_toml(config, pyproject_path)
        print("✅ ruff配置注入完成")
        return True
    
    else:
        print(f"未找到pyproject.toml，创建独立配置文件")
        
        if ruff_toml_path.exists():
            print("✅ .ruff.toml已存在，跳过创建")
            return True
        
        config = get_default_ruff_config(
            line_length=line_length,
            python_version=python_version
        )
        
        save_toml(config, ruff_toml_path)
        print(f"✅ 已创建: {ruff_toml_path}")
        return True


def inject_mypy_config(
    project_root: Path,
    python_version: str = "3.12"
) -> bool:
    """
    注入mypy配置到项目（可选）
    
    Args:
        project_root: 项目根目录
        python_version: Python版本
    
    Returns:
        是否成功
    """
    print("=== mypy配置注入 ===")
    
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("⚠️  未找到pyproject.toml，跳过mypy配置")
        return False
    
    config = load_toml(pyproject_path)
    
    if has_mypy_config(config):
        print("✅ 已有mypy配置，跳过注入")
        return True
    
    print("注入mypy配置到pyproject.toml...")
    
    config.setdefault("tool", {})
    config["tool"]["mypy"] = get_default_mypy_config(python_version=python_version)
    
    save_toml(config, pyproject_path)
    print("✅ mypy配置注入完成")
    return True


if __name__ == "__main__":
    import sys
    
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    
    inject_ruff_config(project_root)
    print()
    inject_mypy_config(project_root)
