#!/usr/bin/env python3
"""便捷启动脚本"""
import sys
from pathlib import Path

skill_src = Path(__file__).parent / "src"
sys.path.insert(0, str(skill_src))

from python_codecheck.main import main

if __name__ == "__main__":
    sys.exit(main())
