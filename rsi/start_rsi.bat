@echo off
rem RSI 一键启动：SME(8000) + daemon(8100)。工作目录=本脚本所在目录的上一级。
chcp 65001 >nul
set RSI_ROOT=%~dp0..
cd /d "%RSI_ROOT%"
set PY=%RSI_ROOT%\.venv\Scripts\python.exe

echo [1/2] 启动 SME 记忆服务 (http://127.0.0.1:8000) ...
start "SME" /min "%PY%" -m sme.api --config rsi\configs\sme.rsi.json
timeout /t 5 /nobreak >nul

echo [2/2] 启动 RSI daemon (http://127.0.0.1:8100) ...
start "RSI-daemon" /min "%PY%" -m rsi.daemon.main --config rsi\configs\daemon.json

echo 完成。SME: http://127.0.0.1:8000  daemon: http://127.0.0.1:8100/status
echo 真跑模式请先设置环境变量 RSI_LLM_API_KEY（智谱/DeepSeek 等 OpenAI 兼容密钥）。
