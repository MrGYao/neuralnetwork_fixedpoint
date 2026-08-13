"""配置模板"""
from typing import Dict, Any, List


def get_default_ruff_config(
    line_length: int = 100,
    python_version: str = "py310",
    known_first_party: List[str] = None
) -> Dict[str, Any]:
    """
    获取默认ruff配置
    
    Args:
        line_length: 行长度限制
        python_version: Python版本
        known_first_party: 已知的第一方包名
    
    Returns:
        ruff配置字典
    """
    if known_first_party is None:
        known_first_party = []
    
    config = {
        "line-length": line_length,
        "target-version": python_version,
        "exclude": [
            "__pycache__",
            ".venv",
            "node_modules",
            ".git",
            "*.egg-info",
            ".tox",
            ".eggs",
        ],
        "lint": {
            "select": [
                "E",      # pycodestyle errors
                "W",      # pycodestyle warnings
                "F",      # Pyflakes
                "I",      # isort
                "B",      # flake8-bugbear
                "C4",     # flake8-comprehensions
                "UP",     # pyupgrade
                "ARG",    # flake8-unused-arguments
                "SIM",    # flake8-simplify
            ],
            "ignore": [
                "E501",   # line too long (交给formatter处理)
                "B008",   # function call in default argument
                "W191",   # indentation contains tabs
            ],
        },
        "format": {
            "quote-style": "double",
            "indent-style": "space",
            "skip-magic-trailing-comma": False,
        }
    }
    
    if known_first_party:
        config["lint"]["isort"] = {
            "known-first-party": known_first_party
        }
    
    return config


def get_default_mypy_config(
    python_version: str = "3.12"
) -> Dict[str, Any]:
    """
    获取默认mypy配置
    
    Args:
        python_version: Python版本
    
    Returns:
        mypy配置字典
    """
    return {
        "python_version": python_version,
        "warn_return_any": True,
        "warn_unused_configs": True,
        "disallow_untyped_defs": False,
        "ignore_missing_imports": True,
        "show_error_codes": True,
        "pretty": True,
    }
