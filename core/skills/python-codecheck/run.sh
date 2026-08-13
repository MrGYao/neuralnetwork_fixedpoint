#!/usr/bin/env bash
# Unix便捷启动脚本
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
python -m python_codecheck.main "$@"
