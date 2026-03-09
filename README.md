# TinySwallow Dashboard

ESP32ロボットと通信し、YOLOによる物体検出およびLLMによる自然言語指示を用いて制御を行うWebダッシュボードシステムです。
カメラ映像をWebブラウザに配信し、対象物の検出結果をもとに自動追従や制御を行います。

---

# システム概要

本システムは以下の機能を提供します。

* ESP32とのUDP通信によるロボット制御
* カメラ画像の受信および配信
* YOLOによる物体検出
* TinySwallow LLMによる自然言語指示処理
* Webダッシュボードによる遠隔操作
* 自律走行アルゴリズムによるロボット制御

---

# 動作環境

## 使用言語

* Python 3.10

## 主なライブラリ

| ライブラリ            | 用途      |
| ---------------- | ------- |
| FastAPI          | Webサーバ  |
| Uvicorn          | ASGIサーバ |
| OpenCV           | 画像処理    |
| Ultralytics YOLO | 物体検出    |
| llama-cpp-python | LLM推論   |
| HuggingFace Hub  | モデル取得   |

依存関係は `requirements.txt` に記述されています。

---

# セットアップ

## 1. リポジトリの取得

```bash
git clone https://github.com/Yushin404/torripi4-Python.git
cd tinyswallow_dashboard
```

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

# 検出対象（YOLO）

YOLOモデルでは以下の3種類の物体を検出します。

| クラスID | 物体    | 用途      |
| ----- | ----- | ------- |
| 0     | clip  | 回収対象    |
| 1     | cone  | 障害物     |
| 2     | stool | clipの目印 |

clipはstoolの下に配置されるため、
clipが見えない場合は **stoolを目印として接近**します。

---

# 検出安定化アルゴリズム

物体検出は1フレーム単位では誤検出や検出消失が発生するため、
本システムでは **時間フィルタ（Temporal Filtering）** を導入しています。

## 検出確定条件

同一物体が複数フレーム連続で検出された場合のみ有効とします。

| 物体    | 必要フレーム |
| ----- | ------ |
| clip  | 2      |
| stool | 2      |
| cone  | 1      |

clipとstoolは誤検出防止のため2フレーム連続検出を必要とします。
coneは障害物であるため1フレームで即反応します。

---

## 検出保持

一時的に検出が消えた場合でも、数フレームは前回の検出結果を保持します。

| 物体    | 保持フレーム |
| ----- | ------ |
| clip  | 3      |
| stool | 3      |
| cone  | 1      |

これによりYOLOの検出揺らぎによる挙動の不安定化を防ぎます。

---

# 自動運転アルゴリズム

ロボットは以下の優先順位で行動を決定します。

```
cone回避
↓
clip回収
↓
stool追従
↓
探索
```

---

## 1. cone回避

cone（赤コーン）が画面中央付近に検出された場合、衝突を避けるため回避行動を行います。

```
cone右 → 左回転 (A)
cone左 → 右回転 (D)
```

---

## 2. clip回収

clipはロボットの主要な回収対象です。

```
clip中央 → 前進 (W)
clip左   → 左回転 (A)
clip右   → 右回転 (D)
```

clipが中央付近に入った場合、**0.5秒前進**することで回収成功率を高めています。

---

## 3. stool追従

clipが検出されない場合は、clipの設置位置であるstoolへ接近します。

```
stool中央 → 前進
stool左   → 左回転
stool右   → 右回転
```

clipが検出された時点で **clip追跡モードに切り替え**ます。

---

## 4. 探索モード

対象物が検出されない場合は環境探索を行います。

```
右回転 (D)
```

これにより視界内に対象物を探します。

---

# スタック防止機構

ロボットが同じ場所で停止し続けることを防ぐため、
一定時間ごとにリカバリ動作を行います。

10秒ごとに

```
後退 (S)
↓
回転 (D)
```

を実行します。

---

# プロジェクト構成（2026/03時点）

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
│ └ drivers/       # カメラ・モック制御
│
├ templates/
│ └ dashboard.html # WebダッシュボードUI
│
├ static/
│ ├ app.js         # フロントエンドスクリプト
│ └ styles.css     # UIスタイル
│
└ models/
  ├ stools4-11s.pt
  ├ stools5-11s.pt
  └ stools6-11s.pt
```

---

# 主要モジュール

| ファイル         | 説明                    |
| ------------ | --------------------- |
| main.py      | FastAPIサーバおよびシステム全体制御 |
| state.py     | システム状態の共有管理           |
| esp32_udp.py | ESP32とのUDP通信          |
| vision.py    | YOLOによる物体検出と検出安定化     |
| llm.py       | TinySwallowによる自然言語制御  |

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
      ├ Vision Stabilization
      ├ Autonomous Navigation
      ├ LLM Command Parser
      │
      ▼
Web Dashboard (Browser)
```

---
