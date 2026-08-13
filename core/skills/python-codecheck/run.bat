@echo off
REM Windows便捷启动脚本
SET PYTHONPATH=%~dp0src;%PYTHONPATH%
python -m python_codecheck.main %*
