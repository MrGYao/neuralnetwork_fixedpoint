"""查找项目根目录"""
from pathlib import Path
from typing import Optional


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    从给定路径向上查找项目根目录
    
    查找逻辑：
    1. 向上查找包含pyproject.toml且有[project]或[dependency-groups]的目录
    2. 如果未找到，返回Git根目录
    3. 如果仍找不到，返回当前工作目录
    
    Args:
        start_path: 起始路径，默认为当前目录
    
    Returns:
        项目根目录路径
    """
    if start_path is None:
        start_path = Path.cwd()
    
    current = start_path.resolve()
    
    if current.is_file():
        current = current.parent
    
    while current != current.parent:
        pyproject = current / "pyproject.toml"
        
        if pyproject.exists():
            content = pyproject.read_text(encoding="utf-8")
            if "[project]" in content or "[dependency-groups]" in content:
                return current
        
        current = current.parent
    
    from .utils import get_git_root
    git_root = get_git_root()
    if git_root:
        return git_root
    
    return Path.cwd()


def find_project_root_for_file(file_path: Path) -> Path:
    """
    为指定文件查找其所属的项目根目录
    
    Args:
        file_path: 文件路径
    
    Returns:
        项目根目录
    """
    return find_project_root(file_path)


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    root = find_project_root(path)
    print(root)
