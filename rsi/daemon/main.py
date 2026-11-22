"""入口：python -m rsi.daemon.main [--config rsi/configs/daemon.json]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn

from .engine import RSI_BASE, RSIDaemon
from .server import create_app


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    # 测试通道：环境变量强制 provider=fake（不打招呼就切换是被禁止的，仅此一个显式开关）
    if os.environ.get("RSI_TEST_MODE") == "1":
        cfg.setdefault("llm", {})["provider"] = "fake"
        cfg.setdefault("self_propose", {})["enabled"] = False
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent
                                                 / "configs" / "daemon.json"))
    parser.add_argument("--data-dir", type=Path, default=RSI_BASE,
                        help="运行数据目录（dataset、logs、biodata）")
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    daemon = RSIDaemon(cfg, args.data_dir.resolve())
    app = create_app(daemon)
    uvicorn.run(app, host=cfg.get("daemon", {}).get("host", "127.0.0.1"),
                port=int(cfg.get("daemon", {}).get("port", 8100)), log_level="info")


if __name__ == "__main__":
    main()
