# TinySwallow Dashboard

ESP32ロボットと通信し、YOLOによる物体検出およびLLMによる自然言語指示を用いて制御を行うWebダッシュボードシステムです。  
カメラ映像をWebブラウザに配信し、対象物の検出結果をもとに自動追従や制御を行います。

---

# システム概要

本システムは以下の機能を提供します。

- ESP32とのUDP通信によるロボット制御
- カメラ画像の受信および配信
- YOLOによる物体検出
- TinySwallow LLMによる自然言語指示処理
- Webダッシュボードによる遠隔操作

---

# 動作環境

## 使用言語

- Python 3.10

## 主なライブラリ

| ライブラリ | 用途 |
|---|---|
| FastAPI | Webサーバ |
| Uvicorn | ASGIサーバ |
| OpenCV | 画像処理 |
| Ultralytics YOLO | 物体検出 |
| llama-cpp-python | LLM推論 |
| HuggingFace Hub | モデル取得 |

依存関係は `requirements.txt` に記述されています。

---

# セットアップ

## 1. リポジトリの取得

```bash
git clone https://github.com/Yushin404/torripi4-Python.git
cd tinyswallow_dashboard
````

## 2. 必要ライブラリのインストール

```bash
pip install -r requirements.txt
```

---

# サーバ起動

以下のコマンドでFastAPIサーバを起動します。

```bash
./run.sh
```

または

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# アクセス方法

ブラウザで以下のURLにアクセスしてください。

```
http://localhost:8000
```

ダッシュボード画面が表示され、カメラ映像およびロボット制御が可能になります。

---

# 他デバイスからのアクセス

サーバPCのファイアウォールが無効化されている場合、同一WiFiネットワーク上の他のデバイスからもアクセス可能です。

その場合は `localhost` の代わりにサーバPCのIPアドレスを指定します。

例：

```
http://192.168.1.xxx:8000
```

スマートフォンや別PCからダッシュボードを閲覧することができます。

---

# プロジェクト構成

```
tinyswallow_dashboard/
│
├ requirements.txt
│  # システムの実行に必要なPythonライブラリ
│
├ run.sh
│  # FastAPIサーバ起動スクリプト（MacOS / Linux）
│
├ app/
│ ├ main.py        # FastAPIサーバ・システム制御
│ ├ state.py       # グローバル状態管理
│ ├ esp32_udp.py   # ESP32とのUDP通信
│ ├ vision.py      # YOLO物体検出
│ ├ llm.py         # TinySwallowによる自然言語制御
│ └ __pycache__/
│
├ templates/
│ └ dashboard.html # WebダッシュボードUI
│
├ static/
│ ├ app.js         # フロントエンドスクリプト
│ └ styles.css     # UIスタイル
│
└ models/
  ├ stools4-11s.pt # YOLOモデル
  ├ stools5-11s.pt # YOLOモデル
  └ stools6-11s.pt # YOLOモデル
```

---

# 主要モジュール

| ファイル         | 説明                     |
| ------------ | ---------------------- |
| main.py      | FastAPIサーバおよびシステム全体の制御 |
| state.py     | システムの状態情報を共有管理         |
| esp32_udp.py | ESP32とのUDP通信処理         |
| vision.py    | YOLOによる物体検出            |
| llm.py       | TinySwallowによる自然言語指示処理 |

---

# システム構成

```
ESP32 (Camera)
      │
      │ UDP
      ▼
Python Server (FastAPI)
      │
      ├ YOLO Object Detection
      ├ LLM Command Parser
      │
      ▼
Web Dashboard (Browser)
```

---

# ライセンス

MIT License
