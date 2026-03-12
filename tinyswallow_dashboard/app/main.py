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

import os
from .drivers.droidcam import DroidCamReader
from .drivers.mock_commands import MockCommandSink


templates = Jinja2Templates(directory="templates")
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

cmd_sink = MockCommandSink()
droidcam = None


# =======================
# あなたの環境のIP設定
# =======================
LOCAL_UDP_IP = "0.0.0.0"
SHARED_UDP_PORT = 50000
ESP32_UDP_IP = "192.168.1.1"
ESP32_UDP_PORT = 55555

# 自動運転パラメータ（あなたの2本目に合わせたもの）
CENTER_THRESHOLD = 60
SEND_INTERVAL = 0.15
FRAME_CENTER_X = 240 // 2

# =======================
# 初期化（起動時に一回だけ）
# =======================
cmd_sink = MockCommandSink()
droidcam = None
esp = None

vision = Vision(model_path="models/stools7-11s.pt", priority_class_ids=[0, 2])
llm = build_llm()


receiver_thread = None
infer_thread = None

@app.on_event("startup")
async def on_startup():
    global esp, cmd_sink

    #本番(ESP)はこっち
    esp = Esp32Udp("0.0.0.0", 50000, "192.168.1.1", 55555)
    start_receiver(esp)
    cmd_sink = esp

    
    # # テスト（Droidcam）を使っているときは以下のコード
    # global droidcam
    # # DroidCam URL（環境変数で差し替え可）
    # url = os.getenv("DROIDCAM_URL", "http://192.168.11.38:4747/video")
    # droidcam = DroidCamReader(url=url, target_size=(240, 240))
    # droidcam.start()

    vision.start_infer_loop(fps_limit=15.0)
    asyncio.create_task(auto_control_loop())



@app.on_event("shutdown")
async def on_shutdown():
    STATE.running = False
    vision.stop()
    if droidcam is not None:
        droidcam.stop()



@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/status")
def status():
    return JSONResponse({
        "auto_enabled": STATE.auto_enabled,
        "reverse_flag": STATE.reverse_flag,
        "last_cmd": STATE.last_cmd,

        "clip_detected": STATE.clip_detected,
        "clip_center_x": STATE.clip_center_x,
        "clip_size": STATE.clip_size,

        "cone_detected": STATE.cone_detected,
        "cone_center_x": STATE.cone_center_x,
        "cone_size": STATE.cone_size,

        "stool_detected": STATE.stool_detected,
        "stool_center_x": STATE.stool_center_x,
        "stool_size": STATE.stool_size,

        "target_detected": STATE.target_detected,
        "target_center_x": STATE.target_center_x,
        "target_class_id": STATE.target_class_id,
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
            msg_type = data.get("type")

            if msg_type == "manual":
                cmd = str(data.get("cmd", "")).upper()

                # 自動中は手動を無効化（必要ならこのまま）
                if not STATE.auto_enabled:
                    cmd_sink.send_command(cmd)

                await ws.send_json({"ok": True})

            elif msg_type == "auto":
                STATE.auto_enabled = bool(data.get("enabled", False))
                await ws.send_json({"ok": True, "auto_enabled": STATE.auto_enabled})

            elif msg_type == "reverse":
                STATE.reverse_flag = bool(data.get("enabled", True))
                await ws.send_json({"ok": True, "reverse_flag": STATE.reverse_flag})

            elif msg_type == "nl":
                text = str(data.get("text", "")).strip()
                if not text:
                    await ws.send_json({"ok": False, "error": "empty text"})
                    continue

                try:
                    act = nl_to_action(llm, text)
                    print("[NL]", text, "=>", act)

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
                            cmd_sink.send_command(act["cmd"])

                    await ws.send_json({"ok": True, "parsed": act, "auto_enabled": STATE.auto_enabled})

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    await ws.send_json({"ok": False, "error": repr(e)})

            else:
                await ws.send_json({"ok": False, "error": "unknown type"})

    except WebSocketDisconnect:
        return


# ---------- 自動運転ループ ----------
async def auto_control_loop():
    last_send = 0.0
    last_reverse = 0.0

    REVERSE_INTERVAL = 10.0
    REVERSE_DURATION = 0.75
    ROTATE_DURATION = 0.3

    CLIP_FORWARD_TIME = 0.5
    CONE_CENTER_THRESHOLD = 50
    STOOL_CENTER_THRESHOLD = CENTER_THRESHOLD
    CLIP_CENTER_THRESHOLD = CENTER_THRESHOLD

    while STATE.running:
        await asyncio.sleep(0.01)

        if not STATE.auto_enabled:
            continue

        now = time.time()

        # 定期バック
        if now - last_reverse > REVERSE_INTERVAL:
            cmd_sink.send_command("S")
            await asyncio.sleep(REVERSE_DURATION)

            cmd_sink.send_command("D")
            await asyncio.sleep(ROTATE_DURATION)

            last_reverse = time.time()
            last_send = last_reverse
            continue

        if now - last_send < SEND_INTERVAL:
            continue

        cone_detected = STATE.cone_detected
        cone_center_x = STATE.cone_center_x

        clip_detected = STATE.clip_detected
        clip_center_x = STATE.clip_center_x

        stool_detected = STATE.stool_detected
        stool_center_x = STATE.stool_center_x

        # 1) cone回避
        if cone_detected and cone_center_x is not None:
            cone_diff = cone_center_x - FRAME_CENTER_X

            if abs(cone_diff) <= CONE_CENTER_THRESHOLD:
                if cone_diff >= 0:
                    cmd_sink.send_command("A")
                else:
                    cmd_sink.send_command("D")

                last_send = now
                continue

        # 2) clip優先
        if clip_detected and clip_center_x is not None:
            clip_diff = clip_center_x - FRAME_CENTER_X

            if abs(clip_diff) <= CLIP_CENTER_THRESHOLD:
                cmd_sink.send_command("W")
                await asyncio.sleep(CLIP_FORWARD_TIME)
            else:
                if clip_diff > 0:
                    cmd_sink.send_command("D")
                else:
                    cmd_sink.send_command("A")

            last_send = time.time()
            continue

        # 3) stool追従
        if stool_detected and stool_center_x is not None:
            stool_diff = stool_center_x - FRAME_CENTER_X

            if abs(stool_diff) <= STOOL_CENTER_THRESHOLD:
                cmd_sink.send_command("W")
            else:
                if stool_diff > 0:
                    cmd_sink.send_command("D")
                else:
                    cmd_sink.send_command("A")

            last_send = now
            continue

        # 4) 探索
        cmd_sink.send_command("D")
        last_send = now
