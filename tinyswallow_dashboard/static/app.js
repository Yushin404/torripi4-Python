const ws = new WebSocket(`ws://${location.host}/ws`);

const badgeAuto = document.getElementById("badge-auto");
const badgeCmd = document.getElementById("badge-cmd");
const statusEl = document.getElementById("status");

const reverse = document.getElementById("reverse");
const autoOn = document.getElementById("autoOn");
const autoOff = document.getElementById("autoOff");
const stopBtn = document.getElementById("stop");

const nl = document.getElementById("nl");
const sendNl = document.getElementById("sendNl");

function send(obj) {
  if (ws.readyState === 1) ws.send(JSON.stringify(obj));
}

ws.onmessage = (ev) => {
  // 返答が欲しい時だけ使う（今はほぼOKだけ）
};

async function refreshStatus() {
  const res = await fetch("/status");
  const s = await res.json();
  badgeAuto.textContent = `AUTO: ${s.auto_enabled ? "ON" : "OFF"}`;
  badgeCmd.textContent = `CMD: ${s.last_cmd || "-"}`;
  statusEl.textContent = JSON.stringify(s, null, 2);
}

setInterval(refreshStatus, 500);
refreshStatus();

reverse.addEventListener("change", () => {
  send({type:"reverse", enabled: reverse.checked});
});

autoOn.addEventListener("click", () => send({type:"auto", enabled:true}));
autoOff.addEventListener("click", () => send({type:"auto", enabled:false}));
stopBtn.addEventListener("click", () => send({type:"manual", cmd:"STOP"}));

document.querySelectorAll("button.cmd").forEach(btn => {
  btn.addEventListener("click", () => {
    send({type:"manual", cmd: btn.dataset.cmd});
  });
});

sendNl.addEventListener("click", () => {
  const text = nl.value.trim();
  if (!text) return;
  send({type:"nl", text});
});

// ===== WASD キーボード操作（ブラウザ） =====
// AUTOがONの時はサーバ側が無視する設計
const keyMap = {
  "w": "W",
  "a": "A",
  "s": "S",
  "d": "D",
  "r": "R",
  "g": "G",
  "t": "T",
  "b": "B",
  " ": "STOP",
};

let lastKeySent = 0;
const intervalMs = 80; // 送信レート制限（だいたい12.5Hz）

window.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (!(k in keyMap)) return;

  const now = Date.now();
  if (now - lastKeySent < intervalMs) return;
  lastKeySent = now;

  send({type:"manual", cmd: keyMap[k]});
});
