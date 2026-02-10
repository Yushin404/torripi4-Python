# torripi4-Python

ダッシュボードを作ってみた&SLMを使って日本語による制御ができないかを試してみたい

フォルダ構成
tinyswallow_dashboard/
│
├ README.md
├ requirements.txt
├ run.sh
│
├ app/
│ ├ main.py          # FastAPI サーバ
│ ├ state.py         # グローバル状態管理
│ ├ esp32_udp.py     # UDP通信
│ ├ vision.py        # YOLO推論
│ ├ llm.py           # TinySwallow制御
│
├ templates/
│ └ dashboard.html   # Web UI
│
├ static/
│ ├ app.js
│ └ styles.css
│
├ models/
│ ├ yolo/
│ │ └ stool_best.pt
│ │
│ └ llm/
│   └ tinyswallow.gguf
