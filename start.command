#!/bin/bash
# 個人情報マスキングツール 起動スクリプト（ダブルクリック or ターミナルで ./start.sh）
cd "$(dirname "$0")"
if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg がありません。先に:  brew install ffmpeg"; exit 1
fi
if [ ! -d .venv ]; then
  echo "初回セットアップ中（数分かかります）…"
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt || exit 1
fi
.venv/bin/python app.py
