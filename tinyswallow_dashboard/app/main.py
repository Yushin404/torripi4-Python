from __future__ import annotations
import time
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import cv2

from .state import STATE
from .esp32_udp import Esp32Udp, start_receiver
from .vision import Vision, bgr_to_jpeg_bytes
from .llm import build_llm, nl_to_action

templates = Jinja2Templates(directory="templates")
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# =======================
# あなたの環境のIP設定
# =======================
LOCAL_UDP_IP = "0.0.0.0"
SHARED_UDP_PORT = 50000
ESP32_UDP_IP = "192.168.1.1"
ESP32_UDP_PORT = 55555

# 自動運転パラメータ（あなたの2本目に合わせたもの）
CENTER_THRESHOLD = 80
SEND_INTERVAL = 0.1
FRAME_CENTER_X = 240 // 2

# =======================
# 初期化（起動時に一回だけ）
# =======================
esp = Esp32Udp(LOCAL_UDP_IP, SHARED_UDP_PORT, ESP32_UDP_IP, ESP32_UDP_PORT)
vision = Vision(model_path="models/stools4-11s.pt", target_class_id=0)
llm = build_llm()

receiver_thread = None
infer_thread = None

@app.on_event("startup")
async def on_startup():
    global receiver_thread, infer_thread
    receiver_thread = start_receiver(esp)
    infer_thread = vision.start_infer_loop(fps_limit=15.0)

    # 自動運転ループをバックグラウンドで回す
    asyncio.create_task(auto_control_loop())


@app.on_event("shutdown")
async def on_shutdown():
    STATE.running = False
    vision.stop()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/status")
def status():
    return JSONResponse({
        "auto_enabled": STATE.auto_enabled,
        "reverse_flag": STATE.reverse_flag,
        "last_cmd": STATE.last_cmd,
        "target_detected": STATE.target_detected,
        "target_center_x": STATE.target_center_x,
        "telemetry": STATE.telemetry,
    })


@app.post("/reset")
def reset():
    STATE.auto_enabled = False
    STATE.last_cmd = ""
    return JSONResponse({"ok": True})


# ---------- MJPEG映像 ----------
def mjpeg_generator():
    while STATE.running:
        with STATE.image_lock:
            frame = STATE.image.copy()

        if STATE.reverse_flag:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # 推論スレッドに最新フレームを渡す（推論は別スレッド）
        vision.set_latest_frame(frame)

        annotated = vision.get_latest_annotated()
        if annotated is None:
            # 初回は生フレームを表示
            annotated = frame

        jpg = bgr_to_jpeg_bytes(annotated, 500, 500)
        if not jpg:
            time.sleep(0.01)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.03)  # だいたい ~30fps 上限

@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------- WebSocket制御 ----------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            # data例:
            # {"type":"manual","cmd":"W"}
            # {"type":"auto","enabled":true}
            # {"type":"reverse","enabled":true}
            # {"type":"nl","text":"自動運転オンにして"}
            msg_type = data.get("type")

            if msg_type == "manual":
                cmd = str(data.get("cmd","")).upper()
                # 自動中は手動を無効化したい場合はここで弾く
                if not STATE.auto_enabled:
                    esp.send_command(cmd)
                await ws.send_json({"ok": True})

            elif msg_type == "auto":
                STATE.auto_enabled = bool(data.get("enabled", False))
                await ws.send_json({"ok": True, "auto_enabled": STATE.auto_enabled})

            elif msg_type == "reverse":
                STATE.reverse_flag = bool(data.get("enabled", True))
                await ws.send_json({"ok": True, "reverse_flag": STATE.reverse_flag})

            elif msg_type == "nl":
                text = str(data.get("text","")).strip()
                act = nl_to_action(llm, text)

                # 実行
                if act["action"] == "auto_on":
                    STATE.auto_enabled = True
                elif act["action"] == "auto_off":
                    STATE.auto_enabled = False
                elif act["action"] == "set_reverse_on":
                    STATE.reverse_flag = True
                elif act["action"] == "set_reverse_off":
                    STATE.reverse_flag = False
                elif act["action"] == "send":
                    if not STATE.auto_enabled:
                        esp.send_command(act["cmd"])
                await ws.send_json({"ok": True, "parsed": act, "auto_enabled": STATE.auto_enabled})

            else:
                await ws.send_json({"ok": False, "error": "unknown type"})

    except WebSocketDisconnect:
        return


# ---------- 自動運転ループ ----------
async def auto_control_loop():
    last_send = 0.0
    while STATE.running:
        await asyncio.sleep(0.01)

        if not STATE.auto_enabled:
            continue

        now = time.time()
        if now - last_send < SEND_INTERVAL:
            continue

        # あなたの自動制御ロジック（2本目）を移植
        if STATE.target_detected and STATE.target_center_x is not None:
            diff = abs(STATE.target_center_x - FRAME_CENTER_X)
            if diff <= CENTER_THRESHOLD:
                esp.send_command("W")
            else:
                esp.send_command("M")  # 微小回転
        else:
            esp.send_command("M")

        last_send = now
