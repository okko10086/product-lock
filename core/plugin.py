#!/usr/bin/env python3
"""产品一致性控制插件 v1

三层链路（对应"大模型控制小模型去控制视觉模型"）：
  大模型  —— 调用本插件，给目标描述，读验收结论（不碰参数细节）
  小模型  —— 本插件内的 DriftNet，负责建模/预测漂移/给修正方向
  生成器   —— 按参数卡出图，出完由本插件验收（通过通用 CLI 契约接入，不绑定厂商）

子命令：
  build   产品图 -> 部件级产品模型 model.json
  card    model.json + 目标描述 -> 生成参数卡 card.json（含验收阈值，同源同卡）
  verify  model.json + 生成图 -> 验收报告；--no-model 退回规则层
  chain   一次跑完 build + card，输出给视觉模型用的请求体
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from product_fp import imread_u, product_mask, fingerprint
import gate

PARTS = ["red_cord", "gold_bead", "jade_part"]
PART_LABEL = {"red_cord": "true red braided cord",
              "gold_bead": "polished gold metal beads",
              "jade_part": "pale jade-green carved pendant"}
TOL_EQ = 12.0
TOL_OK = 20.0

_MODEL_DIR = Path(__file__).parent / "训练模型"
_DRIFT = None


def _load_drift():
    """懒加载训练好的 DriftNet。缺失时返回 None，自动退回规则层。"""
    global _DRIFT
    if _DRIFT is False:
        return None
    if _DRIFT is not None:
        return _DRIFT
    try:
        import torch
        from train_driftnet import DriftNet
        w = _MODEL_DIR / "driftnet_v0.pt"
        if not w.exists():
            _DRIFT = False
            return None
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        net = DriftNet(24).to(dev)
        net.load_state_dict(torch.load(w, map_location=dev))
        net.eval()
        _DRIFT = (net, dev)
        return _DRIFT
    except Exception:
        _DRIFT = False
        return None


def _hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0


def _part_sel(h, s, v, name):
    if name == "red_cord":
        return ((h < 14) | (h >= 344)) & (s > 0.38)
    if name == "gold_bead":
        return (h >= 14) & (h < 40) & (s > 0.35) & (v > 0.45)
    return (h >= 40) & (h < 95) & (s > 0.10) & (v > 0.35)


def _part_stats(bgr, sel):
    if int(sel.sum()) < 150:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0][sel] / 255 * 100
    a = lab[..., 1][sel] - 128
    b = lab[..., 2][sel] - 128
    return {"share": round(float(sel.sum()) / (bgr.shape[0] * bgr.shape[1]), 4),
            "L": round(float(L.mean()), 2), "a": round(float(a.mean()), 2), "b": round(float(b.mean()), 2)}


def build_model(img_path: str) -> dict:
    bgr = imread_u(Path(img_path))
    if bgr is None:
        raise SystemExit(f"读不出图: {img_path}")
    mask = product_mask(bgr)
    h, s, v = _hsv(bgr)
    m = mask.astype(bool)
    parts = {}
    for name in PARTS:
        sel = m & _part_sel(h, s, v, name)
        sel = cv2.morphologyEx(sel.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        st = _part_stats(bgr, sel)
        if st:
            parts[name] = st
    fp = fingerprint(bgr, mask)
    return {"source": str(img_path), "size": [int(bgr.shape[1]), int(bgr.shape[0])],
            "product_share": round(float(mask.mean()), 4),
            "overall": {k: fp[k] for k in ("L_mean", "a_mean", "b_mean", "texture_energy", "edge_density")} if fp else {},
            "parts": parts}


def make_card(model: dict, target: str = "") -> dict:
    desc = [PART_LABEL[k] for k in PARTS if k in model["parts"]]
    prompt = (f"commercial product photograph of {' and '.join(desc)}, "
              "on pure white seamless background, soft studio softbox lighting from upper left, "
              "subtle contact shadow, shot on a full-frame camera with a 100mm macro lens at f/8, "
              "ISO 100, high resolution, true-to-life color, natural micro texture")
    if target:
        prompt += f", {target}"
    return {
        "card_id": f"card_{abs(hash(model['source'])) % 10**8:08d}",
        "source_model": model["source"],
        "prompt": prompt,
        "negative": ("deformed product, changed material, altered texture, color shift, plastic look, "
                     "CGI, 3D render, over-sharpened, over-smooth, blown highlights, text, watermark"),
        "locked_parts": [k for k in PARTS if k in model["parts"]],
        "acceptance": {"tol_eq": TOL_EQ, "tol_ok": TOL_OK,
                       "basis": "部件级 Lab 偏移；暂定值，待同源样本标定",
                       "parts": {k: {"L": v["L"], "a": v["a"], "b": v["b"], "share": v["share"]}
                                 for k, v in model["parts"].items()}},
        "degraded_fields": ["reference_strength", "seed", "steps", "cfg"],
    }


def _predict_by_model(model: dict, gen_path: str):
    """小模型预测：返回 3 部件 × (ΔL,Δa,Δb)。"""
    got = _load_drift()
    if got is None:
        return None
    import torch
    from train_driftnet import normalize_pair
    net, dev = got
    ref = imread_u(Path(model["source"]))
    gen = imread_u(Path(gen_path))
    if ref is None or gen is None:
        return None
    pair = normalize_pair(ref, gen)
    if pair is None:
        return None
    x = torch.from_numpy(pair).permute(2, 0, 1).unsqueeze(0).to(dev).float()
    with torch.no_grad():
        out = net(x)[0].detach().cpu().numpy()
    return {PARTS[i]: {"dL": float(out[i * 3]), "da": float(out[i * 3 + 1]), "db": float(out[i * 3 + 2])}
            for i in range(3)}


def _predict_by_rule(model: dict, gen_path: str):
    bgr = imread_u(Path(gen_path))
    if bgr is None:
        return None
    h, s, v = _hsv(bgr)
    m = product_mask(bgr).astype(bool)
    out = {}
    for name in PARTS:
        if name not in model["parts"]:
            continue
        sel = m & _part_sel(h, s, v, name)
        sel = cv2.morphologyEx(sel.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        st = _part_stats(bgr, sel)
        r = model["parts"][name]
        if st is None:
            out[name] = {"missing": True}
            continue
        out[name] = {"dL": st["L"] - r["L"], "da": st["a"] - r["a"], "db": st["b"] - r["b"]}
    return out


def verify(model: dict, gen_path: str, use_model: bool = False) -> dict:
    pred = _predict_by_model(model, gen_path) if use_model else None
    source = "driftnet_v0" if pred is not None else "rule"
    if pred is None:
        pred = _predict_by_rule(model, gen_path) or {}
    rows, fixes = [], []
    for name, p in pred.items():
        if p.get("missing"):
            rows.append({"part": name, "status": "missing", "delta": None})
            fixes.append({"part": name, "action": "boost", "reason": "生成图里找不到该部件"})
            continue
        d = float(np.sqrt(p["dL"] ** 2 + p["da"] ** 2 + p["db"] ** 2))
        ok = d <= TOL_EQ
        rows.append({"part": name, "delta": round(d, 2),
                     "status": "ok" if ok else ("warn" if d <= TOL_OK else "bad"),
                     "dL": round(p["dL"], 2), "da": round(p["da"], 2), "db": round(p["db"], 2)})
        if not ok:
            direction = []
            if abs(p["da"]) > 6: direction.append("偏红" if p["da"] > 0 else "偏绿")
            if abs(p["db"]) > 6: direction.append("偏黄" if p["db"] > 0 else "偏蓝")
            if abs(p["dL"]) > 6: direction.append("偏亮" if p["dL"] > 0 else "偏暗")
            fixes.append({"part": name, "action": "relock_color",
                          "direction": direction or ["轻微偏移"],
                          "hint": f"只锁 {name} 的颜色词，按方向反向补"})
    passed = bool(rows) and all(r["status"] == "ok" for r in rows)
    return {"generated": str(gen_path), "predictor": source, "passed": passed, "parts": rows, "fixes": fixes}




# ---------------- generate / poster：命令生成器 ----------------
#
# 生成器通过一个通用 CLI 约定接入（见 docs/adapter.md），本插件不认识任何具体厂商：
#   <cli> video  --image PATH --prompt TEXT --duration SEC --ratio W:H --model NAME
#   <cli> image  --image PATH --prompt TEXT --count N --ratio R --size S --model NAME
#   <cli> fetch  --thread ID --run ID --outdir DIR
#   <cli> balance
# 前两条会花钱，必须过守门人（tools/guard）拿一次性令牌。

GUARD = Path(__file__).resolve().parent.parent / "tools" / "guard"
GENERATOR_ENV = "GENERATOR_CLI"


def _guard_cli() -> str:
    """拿生成器 CLI 路径，并确认它指向守门人；指向真 CLI 就直接拒。

    插件内部拦得住自己，拦不住直接调 CLI 出片，所以生成入口必须是守门人。
    """
    import os
    cli = os.environ.get(GENERATOR_ENV) or str(GUARD)
    if Path(cli).name != GUARD.name:
        gate.audit("exec_refuse", reason=f"{GENERATOR_ENV} 未指向守门人", cli=cli)
        raise SystemExit(
            f"闸门拒绝：{GENERATOR_ENV} 指向的是真 CLI，能绕过插件直接出片。\n"
            f"改成守门人再用：export {GENERATOR_ENV}={GUARD}"
        )
    return cli


def _gen_submit(verb: str, args: dict, token: str = "") -> dict:
    """调生成器提交任务，返回解析后的 JSON。失败直接抛，不吞。"""
    import os, subprocess, json as _json
    cli = _guard_cli()
    cmd = [cli, verb]
    for k, v in args.items():
        if v in (None, ""):
            continue
        cmd += [f"--{k}", str(v)]
    env = dict(os.environ)
    if token:
        env["PLUGIN_TOKEN"] = token
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        raise SystemExit(f"提交失败: {(r.stderr or r.stdout)[-400:]}")
    return _json.loads((r.stdout or "").strip().splitlines()[-1])


def _gen_fetch(thread: str, run: str, outdir: str, timeout_s: int = 600, poll_s: int = 10):
    """轮询并下载结果。适配器在完成时打印 {"completed": true}。"""
    import json as _json, subprocess, time
    cli = _guard_cli()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = subprocess.run([cli, "fetch", "--thread", thread, "--run", run, "--outdir", outdir],
                           capture_output=True, text=True)
        out = r.stdout or ""
        try:
            done = _json.loads(out.strip().splitlines()[-1]).get("completed") is True
        except Exception:
            done = '"completed": true' in out or '"completed":true' in out
        if done:
            return sorted(str(x) for x in Path(outdir).iterdir() if x.is_file())
        time.sleep(poll_s)
    raise SystemExit(f"轮询超时（{timeout_s}s）")


def cmd_generate(model_path: str, count: int, outdir: str):
    """card -> 视觉模型 -> 下载结果 -> 逐个验收。"""
    m = json.loads(Path(model_path).read_text(encoding="utf-8"))
    card = make_card(m)
    gate.log_submit("image", model=model_path, count=count, prompt=card.get("prompt", "")[:120])
    token = gate.issue_token_for_image(card, "default", count)
    info = _gen_submit("image", {
        "image": card["source_model"], "prompt": card["prompt"],
        "count": count, "ratio": "3:4", "size": "2K", "model": "default",
    }, token=token)
    gate.log_result("image", True, thread_id=info.get("thread_id"), run_id=info.get("run_id"), count=count)
    files = _gen_fetch(info["thread_id"], info["run_id"], outdir)
    reports = []
    for f in files:
        rep = verify(m, f)
        reports.append({"file": f, "passed": rep["passed"],
                        "parts": [{"part": x["part"], "delta": x["delta"], "status": x["status"]} for x in rep["parts"]],
                        "fixes": rep["fixes"]})
    best = min(reports, key=lambda r: sum(p["delta"] or 999 for p in r["parts"])) if reports else None
    return {"submitted": info, "files": files, "reports": reports,
            "best": best["file"] if best else None,
            "pass_rate": f"{sum(1 for r in reports if r['passed'])}/{len(reports)}"}


def cmd_poster(model_path: str, out: str, title: str = "", subtitle: str = "", tag: str = ""):
    """产品像素直用 + 版式 -> 海报。产品零改动。"""
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    m = json.loads(Path(model_path).read_text(encoding="utf-8"))
    src = m["source"]
    img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"读不出图: {src}")
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32); alpha = img[..., 3].astype(np.float32) / 255.0
    else:
        bgr = img.astype(np.float32); alpha = np.ones(img.shape[:2], np.float32)

    W, H = 1080, 1440
    yy = np.broadcast_to(np.linspace(0, 1, H)[:, None], (H, W))
    gx = np.broadcast_to(np.linspace(0, 1, W)[None, :], (H, W))
    base = np.zeros((H, W, 3), np.float32)
    base[..., 0] = 246 - 6 * yy; base[..., 1] = 246 - 8 * yy; base[..., 2] = 244 - 12 * yy
    glow = np.zeros((H, W), np.float32)
    cv2.circle(glow, (int(W*0.86), int(H*0.13)), int(W*0.34), 1.0, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 180)
    base[..., 0] += glow * 2; base[..., 1] += glow * 6; base[..., 2] += glow * 10
    canvas = np.clip(base, 0, 255).astype(np.uint8)

    x0, y0, x1, y1 = 200, 400, 880, 1180
    ys, xs = np.where(alpha > 0.02)
    py0, py1, px0, px1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = py1-py0+1, px1-px0+1
    scale = min((x1-x0)/pw, (y1-y0)/ph) * 0.94
    nw, nh = max(1, int(pw*scale)), max(1, int(ph*scale))
    prod = cv2.resize(bgr[py0:py1+1, px0:px1+1], (nw, nh), interpolation=cv2.INTER_AREA)
    pa = cv2.resize(alpha[py0:py1+1, px0:px1+1], (nw, nh), interpolation=cv2.INTER_AREA)[..., None]
    ox, oy = x0 + (x1-x0-nw)//2, y0 + (y1-y0-nh)//2
    sh = np.zeros((H, W, 3), np.float32)
    cv2.ellipse(sh, (ox+nw//2, oy+nh-int(nh*0.02)), (int(nw*0.34), int(nh*0.025)), 0, 0, 360, (46,46,50), -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 26)
    canvas = np.clip(canvas.astype(np.float32) - sh*0.55, 0, 255).astype(np.uint8)
    roi = canvas[oy:oy+nh, ox:ox+nw].astype(np.float32)
    canvas[oy:oy+nh, ox:ox+nw] = np.clip(prod*pa + roi*(1-pa), 0, 255).astype(np.uint8)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    def _font(sz, bold=False):
        """跨平台找一款能写中文的字体。

        之前只认 PingFang.ttc，但这台机器上没有这个文件，于是静默退回
        load_default()，中文全变成豆腐块。这里按平台给一串候选，逐个试。
        """
        cands = [
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            # Linux
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            # Windows
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        for c in cands:
            if not Path(c).exists():
                continue
            for idx in (2, 1, 0, None):     # ttc 里可能有多个字面，挨个试
                try:
                    return ImageFont.truetype(c, sz) if idx is None else ImageFont.truetype(c, sz, index=idx)
                except Exception:
                    continue
        return ImageFont.load_default()
    d.text((90, 76), "JADE · HANDCRAFTED", font=_font(24), fill=(120,96,88))
    d.text((90, 150), title or "一抹朱红", font=_font(96, True), fill=(43,33,30))
    d.text((90, 266), subtitle or "守一份东方雅意", font=_font(52), fill=(43,33,30))
    d.line([(90, 350), (210, 350)], fill=(168,74,58), width=3)
    d.text((90, 376), "手工编绳 · 天然玉饰 · 金工点缀", font=_font(28), fill=(120,96,88))
    for i, t in enumerate(["朱砂红编绳，不易褪色", "天然玉料，一物一纹", "金珠手工缠结，牢固耐戴"]):
        cy = 1230 + i*44
        d.ellipse([90, cy+10, 98, cy+18], fill=(168,74,58))
        d.text((112, cy), t, font=_font(26), fill=(70,58,52))
    tg = tag or "新中式 · 手作"
    tf = _font(26); tw = d.textlength(tg, font=tf)
    tx, ty = W-90-tw-44, H-120
    d.rounded_rectangle([tx, ty, tx+tw+44, ty+62], radius=31, outline=(168,74,58), width=2)
    d.text((tx+22, ty+15), tg, font=tf, fill=(168,74,58))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pil.save(out, quality=95)
    return {"poster": str(out), "size": list(pil.size), "product_untouched": True}


def register_extra(sub):
    p5 = sub.add_parser("generate", help="card -> 视觉模型 -> 验收")
    p5.add_argument("--model", required=True)
    p5.add_argument("--count", type=int, default=2)
    p5.add_argument("--outdir", required=True)
    p6 = sub.add_parser("poster", help="产品像素直用 -> 海报")
    p6.add_argument("--model", required=True)
    p6.add_argument("--out", required=True)
    p6.add_argument("--title", default="")
    p6.add_argument("--subtitle", default="")
    p6.add_argument("--tag", default="")


# ---------------- plan / run：剧本 -> 分镜方案 -> 确认 -> 执行 ----------------

# ---------------- 内置词库（最小可用默认集） ----------------
# 完整词库放在 lexicon/lexicon.json，启动时覆盖这里。
# 这里只留几条通用运镜，保证没有 lexicon 文件时 plan 也能跑。
# 说明文字为项目自带，可整体替换。
CAMERA_LIB = {
    "推近": {"en": "Dolly in", "safe": True, "how": "机身沿光轴向前移动靠近主体",
             "param": "3-5 秒推进 8-12%", "risk": "推过量主体边缘出画"},
    "拉远": {"en": "Dolly out", "safe": True, "how": "机身沿光轴后退",
             "param": "4-6 秒后退 10-15%", "risk": "退太多露出布景边缘"},
    "横摇": {"en": "Pan", "safe": True, "how": "机位不动，机身绕竖轴左右转",
             "param": "单次 ≤15°，起停各留 0.3 秒", "risk": "角度大就带出画"},
    "俯仰": {"en": "Tilt", "safe": True, "how": "机位不动，镜头上下点头",
             "param": "单次 ≤10°", "risk": "俯仰过大等于换机位"},
    "变焦推近": {"en": "Zoom in", "safe": True, "how": "机位不动，焦距变长放大主体",
                 "param": "1.2-1.8 秒走完", "risk": "几乎无几何风险"},
    "横移": {"en": "Truck", "safe": True, "how": "机身平行于主体横向平移",
             "param": "速度 ≤15 px/s", "risk": "行程过长露出边界"},
    "拉焦": {"en": "Rack focus", "safe": True, "how": "机位焦距不动，改焦点位置",
             "param": "0.8-1.5 秒完成", "risk": "景深太浅两个都虚"},
    "定格": {"en": "Freeze frame", "safe": True, "how": "画面在某一帧停住",
             "param": "停留 1-2 秒", "risk": "长停像卡了"},
    "环绕": {"en": "Orbit", "safe": False, "how": "机身绕主体画弧",
             "param": "—", "risk": "每个角度都是新产品面，单图做不出来"},
    "跟拍": {"en": "Tracking shot", "safe": False, "how": "机身跟主体同步移动",
             "param": "—", "risk": "主体要整体移动，单图做不出"},
    "推轨变焦": {"en": "Dolly zoom", "safe": False, "how": "后退同时拉长焦距",
                 "param": "—", "risk": "透视持续变化，产品比例会漂"},
    "第一人称穿越": {"en": "FPV fly-through", "safe": False, "how": "镜头穿过场景接近主体",
                     "param": "—", "risk": "途中一直是新角度"},
    "甩镜": {"en": "Whip pan", "safe": False, "how": "极快甩一下靠模糊转场",
             "param": "—", "risk": "整帧在动，无可用帧"},
}
SHOT_LANG_BASIC = {}
SHOT_LANG_SPECIAL = {}
SHOT_LANG_FRAMING = {}
SHOT_LANG_COMPOSITION = {}
EPIC_MOVES = {}
STORY_SHOTS = []
STORY_SHOTS_MORE = []


# 高风险运镜：会改透视或需换视角，产品会变形，方案里一律不用
CAMERA_BANNED = {
    "推轨变焦 Dolly zoom": "机位与焦距反着来，背景必须重绘，图生视频必变形",
    "环绕/轨道": "需要多角度素材，单视角硬转等于贴片挤扁",
    "跟拍": "产品需整体移动，单图无法稳定跟随",
    "摇臂大角度上升": "透视大改，产品跟着改形",
    "第一人称穿越": "透视大改，产品跟着改形",
    "甩镜转场": "动作段速度超稳态线，产品边缘易糊",
}

# 词库里可以整体换成自己的术语表：放一个 lexicon/lexicon.json，启动时覆盖内置默认。
_LEXICON_FILE = Path(__file__).resolve().parent.parent / "lexicon" / "lexicon.json"


def load_lexicon_file(path: Path = _LEXICON_FILE):
    """加载外部词库。文件不在就沿用内置默认，不报错。"""
    global CAMERA_LIB, EPIC_MOVES, STORY_SHOTS, STORY_SHOTS_MORE
    global SHOT_LANG_BASIC, SHOT_LANG_SPECIAL, SHOT_LANG_FRAMING, SHOT_LANG_COMPOSITION
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if isinstance(d.get("运镜"), dict):
        CAMERA_LIB = d["运镜"]
    if isinstance(d.get("史诗"), dict):
        EPIC_MOVES = d["史诗"]
    lang = d.get("镜头语言")
    if isinstance(lang, dict):
        # 四类合并后统一用；分不开就全塞进基础表，一镜一运动主体的约束不受影响
        SHOT_LANG_BASIC = lang
        SHOT_LANG_SPECIAL = {}
        SHOT_LANG_FRAMING = {}
        SHOT_LANG_COMPOSITION = {}
    story = d.get("剧情")
    if isinstance(story, list) and story:
        STORY_SHOTS = story
        STORY_SHOTS_MORE = []
    return True


def lexicon(category: str = "all"):
    """词库总入口。category: 运镜 / 镜头语言 / 史诗 / 剧情 / all"""
    if category == "运镜":
        return CAMERA_LIB
    if category == "镜头语言":
        return {**SHOT_LANG_BASIC, **SHOT_LANG_SPECIAL, **SHOT_LANG_FRAMING, **SHOT_LANG_COMPOSITION}
    if category == "史诗":
        return EPIC_MOVES
    if category == "剧情":
        return {f"{x['no']:02d} {x['name']}": x for x in (STORY_SHOTS + STORY_SHOTS_MORE)}
    return {
        "运镜": CAMERA_LIB,
        "镜头语言": lexicon("镜头语言"),
        "史诗": EPIC_MOVES,
        "剧情": lexicon("剧情"),
    }


# 对产品保真最稳的四个（分镜骨架只用这四个）
SAFE_PRIMARY = ["推近", "变焦推近", "拉焦", "定格"]

# 分镜骨架：只用 safe 运镜，按"一个镜头一个运动主体"排
SHOT_TEMPLATES = [
    {"shot": 1, "cam": "推近", "dur": 4, "subject": "产品整体",
     "first": "Five bracelets on a clean white plinth, wide establishing shot, soft studio light",
     "last":  "Same five bracelets, camera pushed in 10%, still centered, product unchanged"},
    {"shot": 2, "cam": "横摇", "dur": 4, "subject": "背景",
     "first": "Products sharp in foreground, neutral white backdrop",
     "last":  "Products unchanged, backdrop shifted left at one third speed, parallax visible"},
    {"shot": 3, "cam": "定格", "dur": 3, "subject": "光线",
     "first": "Locked camera, macro on braided red cord and jade pendant",
     "last":  "Locked camera, a narrow light band has swept across the jade surface"},
    {"shot": 4, "cam": "变焦推近", "dur": 4, "subject": "产品",
     "first": "Low angle view of the bracelets, product in lower third of frame",
     "last":  "Same low angle, slight perspective lift, product still lower third"},
]

MODEL_TABLE = [
    {"model": "Seedance_2.0_mini_lite", "credit_per_sec": 4, "note": "最快最省，适合样片"},
    {"model": "Seedance_2.5_draft", "credit_per_sec": None, "note": "480P 草稿，先看动态"},
    {"model": "Seedance_2.5", "credit_per_sec": None, "note": "终稿，贵"},
]


def cmd_plan(model_path: str, script: str, out: str, budget: int = 200):
    """剧本/参考 -> 分镜方案（运镜 + 首尾帧提示词 + 推荐手段）。不生成，只出方案等确认。"""
    m = json.loads(Path(model_path).read_text(encoding="utf-8"))
    parts = list(m.get("parts", {}).keys())
    desc = "、".join({"red_cord": "红色编绳", "gold_bead": "金珠", "jade_part": "玉饰"}.get(k, k) for k in parts)

    shots = []
    used = 0
    for t in SHOT_TEMPLATES:
        cam = CAMERA_LIB.get(t["cam"])
        if cam is None or not cam["safe"]:
            continue
        est = t["dur"] * 4
        if used + est > budget:
            break
        used += est
        shots.append({
            "shot_id": t["shot"],
            "duration_s": t["dur"],
            "motion_subject": t["subject"],
            "camera_move": t["cam"],
            "camera_instruction": f"{t['dur']}秒 {cam['en']}，{cam['how']}，{cam['param']}，主体不位移",
            "camera_param": cam["param"],
            "from": "词库",
            "camera_risk": cam["risk"],
            "first_frame_prompt": t["first"],
            "last_frame_prompt": t["last"],
            "recommended_model": "Seedance_2.0_mini_lite",
            "ratio": "3:4",
            "credit_estimate": est,
            "why": "运镜取自词库安全清单；一镜一个运动主体，产品几何不变",
        })

    plan = {
        "script": script or "（未提供，按标准产品展示脚本生成）",
        "product_model": model_path,
        "product_parts": parts,
        "product_desc": desc,
        "shots": shots,
        "total_credits_estimate": used,
        "budget_cap": budget,
        "model_options": MODEL_TABLE,
        "camera_library_size": len(CAMERA_LIB),
        "camera_banned": CAMERA_BANNED,
        "needs_confirm": True,
        "confirm_prompt": "确认分镜与预算后执行 run；要改哪个镜头直接说编号",
        "note": "本方案只锁运镜与首尾帧描述，产品几何由建模保证，生成模型只负责光影与运动。",
    }
    Path(out).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def cmd_run(plan_path: str, outdir: str, confirmed: bool = False, only_shot: int = 0):
    """确认后的方案 -> 逐镜头调生成模型 -> 下载 -> 记录。"""
    import os, subprocess, time
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if not confirmed:
        gate.audit("refuse", reason="未加 --confirm", plan=plan_path,
                   credits=plan.get("total_credits_estimate"))
        raise SystemExit("方案未确认：加 --confirm 表示已人工确认，才会消耗积分")
    gate.require_approved(plan, plan_path)
    _guard_cli()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    done = []
    for s in plan["shots"]:
        if only_shot and s["shot_id"] != only_shot:
            continue
        first_img = plan.get("first_frame_image")
        if not first_img or not Path(first_img).exists():
            gate.audit("refuse", reason="缺首帧图", plan=plan_path, shot=s["shot_id"])
            raise SystemExit(f"镜头 {s['shot_id']} 缺首帧图，方案里需要 first_frame_image")
        gate.log_submit("video", plan=plan_path, shot=s["shot_id"],
                        model=s["recommended_model"], credits=s["credit_estimate"])
        # 签发一次性令牌：只有确认过的方案才签得出来，守门人只认这个令牌
        token = gate.issue_token(plan_path, s["shot_id"], s["recommended_model"], s["credit_estimate"])
        try:
            info = _gen_submit("video", {
                "model": s["recommended_model"], "image": first_img,
                "prompt": s["camera_instruction"] + "，" + s["last_frame_prompt"],
                "ratio": s["ratio"], "duration": s["duration_s"],
            }, token=token)
        except SystemExit as e:
            done.append({"shot": s["shot_id"], "ok": False, "error": str(e)[:300]})
            gate.log_result("video", False, shot=s["shot_id"], error=str(e)[:200])
            continue
        done.append({"shot": s["shot_id"], "ok": True, "thread_id": info["thread_id"],
                     "run_id": info["run_id"], "credits": s["credit_estimate"]})
        gate.log_result("video", True, shot=s["shot_id"], thread_id=info["thread_id"],
                        run_id=info["run_id"], credits=s["credit_estimate"])
        time.sleep(2)
    res = {"outdir": outdir, "submitted": done,
           "total_credits": sum(x.get("credits", 0) for x in done if x.get("ok"))}
    Path(outdir, "_run.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


POSE_DEF = Path(__file__).resolve().parent.parent / "pose" / "poses.json"
CAM_DEF = Path(__file__).resolve().parent.parent / "pose" / "cameras.json"


def load_poses() -> list:
    """读姿态库（9 个基础机位）。路径不对时返回空列表，不炸。"""
    try:
        d = json.loads(POSE_DEF.read_text(encoding="utf-8"))
        return d.get("poses", [])
    except Exception:
        return []


def load_cams() -> list:
    """读机位库（12 个带焦段的机位）。"""
    try:
        d = json.loads(CAM_DEF.read_text(encoding="utf-8"))
        return d.get("shots", [])
    except Exception:
        return []


def cmd_pose(pid: str = "", blender: bool = False, cams: bool = False):
    """姿态库 / 机位库查询。机位库带焦段，姿态库带取景距离。"""
    if cams:
        shots = load_cams()
        if not shots:
            raise SystemExit(f"读不到机位库: {CAM_DEF}")
        if pid:
            shots = [x for x in shots if x["id"].upper() == pid.upper()]
        return {"count": len(shots), "source": str(CAM_DEF),
                "shots": [{"id": x["id"], "name": x["name"], "label": x["label"],
                           "focal": x["focal"], "azimuth": x["azimuth"],
                           "elevation": x["elevation"], "framing": x["framing"],
                           "desc": x.get("desc", "")} for x in shots]}
    poses = load_poses()
    if not poses:
        raise SystemExit(f"读不到姿态库: {POSE_DEF}")
    if pid:
        poses = [p for p in poses if p["id"].upper() == pid.upper()]
        if not poses:
            raise SystemExit(f"没有这个姿态: {pid}")
    if blender:
        return {"poses": [{"id": p["id"], "name": p["name"], "label": p["label"],
                           "azimuth": p["azimuth"], "elevation": p["elevation"],
                           "dist": p["dist"], "fill": p.get("fill", 0.82)} for p in poses]}
    return {"count": len(poses), "source": str(POSE_DEF),
            "poses": [{"id": p["id"], "name": p["name"], "label": p["label"],
                       "azimuth": p["azimuth"], "elevation": p["elevation"],
                       "dist": p.get("dist"), "use": p.get("use", "")} for p in poses]}


def register_plan(sub):
    p7 = sub.add_parser("plan", help="剧本/参考 -> 分镜方案（等确认）")
    p7.add_argument("--model", required=True, help="产品模型 model.json")
    p7.add_argument("--script", default="", help="剧本或参考描述")
    p7.add_argument("--budget", type=int, default=200)
    p7.add_argument("--out", required=True)
    p8 = sub.add_parser("run", help="确认后的方案 -> 执行生成")
    p8.add_argument("--plan", required=True)
    p8.add_argument("--outdir", required=True)
    p8.add_argument("--confirm", action="store_true")
    p8.add_argument("--shot", type=int, default=0, help="只跑某个镜头")
    p9 = sub.add_parser("approve", help="人工确认方案（闸门的放行开关）")
    p9.add_argument("--plan", required=True)
    p9.add_argument("--note", default="")
    p9.add_argument("--by", default="user")
    p11 = sub.add_parser("pose", help="查 3D 姿态库（9 个产品机位）")
    p11.add_argument("--id", default="")
    p11.add_argument("--blender", action="store_true", help="输出 Blender 可直接用的机位表")
    p11.add_argument("--cams", action="store_true", help="查机位库（12 个，带焦段）")
    p10 = sub.add_parser("gate", help="查看闸门状态与最近记录")
    p10.add_argument("--limit", type=int, default=15)


def main():
    load_lexicon_file()
    ap = argparse.ArgumentParser(description="产品一致性控制插件 v1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("build"); p1.add_argument("--image", required=True); p1.add_argument("--out", required=True)
    p2 = sub.add_parser("card"); p2.add_argument("--model", required=True); p2.add_argument("--target", default=""); p2.add_argument("--out", required=True)
    p3 = sub.add_parser("verify"); p3.add_argument("--model", required=True); p3.add_argument("--image", required=True)
    p3.add_argument("--use-model", action="store_true", help="启用 DriftNet 实验层（未通过验证）"); p3.add_argument("--out", default="")
    p4 = sub.add_parser("chain"); p4.add_argument("--image", required=True); p4.add_argument("--target", default=""); p4.add_argument("--outdir", required=True)
    register_extra(sub)
    register_plan(sub)
    pl = sub.add_parser("lexicon", help="查词库")
    pl.add_argument("--category", default="all", choices=["all", "运镜", "镜头语言", "史诗", "剧情"])
    args = ap.parse_args()

    if args.cmd == "build":
        m = build_model(args.image)
        Path(args.out).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "parts": list(m["parts"]), "out": args.out}, ensure_ascii=False))
    elif args.cmd == "card":
        m = json.loads(Path(args.model).read_text(encoding="utf-8"))
        c = make_card(m, args.target)
        Path(args.out).write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "card_id": c["card_id"], "out": args.out}, ensure_ascii=False))
    elif args.cmd == "verify":
        m = json.loads(Path(args.model).read_text(encoding="utf-8"))
        r = verify(m, args.image, use_model=args.use_model)
        if args.out:
            Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"predictor": r["predictor"], "passed": r["passed"],
                          "parts": [{"part": x["part"], "delta": x["delta"], "status": x["status"]} for x in r["parts"]],
                          "fixes": r["fixes"]}, ensure_ascii=False, indent=2))
    elif args.cmd == "lexicon":
        lib = lexicon(args.category)
        if args.category == "all":
            print(json.dumps({k: len(v) for k, v in lib.items()}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(lib, ensure_ascii=False, indent=2))
    elif args.cmd == "plan":
        pl = cmd_plan(args.model, args.script, args.out, args.budget)
        print(json.dumps({"shots": [{"id": s["shot_id"], "move": s["camera_move"],
                                     "dur": s["duration_s"], "credit": s["credit_estimate"]} for s in pl["shots"]],
                          "total_credits": pl["total_credits_estimate"],
                          "needs_confirm": pl["needs_confirm"], "out": args.out},
                         ensure_ascii=False, indent=2))
    elif args.cmd == "run":
        r = cmd_run(args.plan, args.outdir, args.confirm, args.shot)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.cmd == "generate":
        out = cmd_generate(args.model, args.count, args.outdir)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.cmd == "poster":
        out = cmd_poster(args.model, args.out, args.title, args.subtitle, args.tag)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.cmd == "approve":
        pl = gate.approve(args.plan, args.note, args.by)
        print(json.dumps({"ok": True, "approved": True, "approved_at": pl["approved_at"],
                          "plan": args.plan, "credits": pl.get("total_credits_estimate")},
                         ensure_ascii=False, indent=2))
    elif args.cmd == "pose":
        print(json.dumps(cmd_pose(args.id, args.blender, args.cams), ensure_ascii=False, indent=2))
    elif args.cmd == "gate":
        print(json.dumps(gate.status(args.limit), ensure_ascii=False, indent=2))
    elif args.cmd == "chain":
        od = Path(args.outdir); od.mkdir(parents=True, exist_ok=True)
        m = build_model(args.image)
        (od / "model.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        c = make_card(m, args.target)
        (od / "card.json").write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "outdir": str(od), "parts": list(m["parts"]),
                          "prompt": c["prompt"][:120] + "...",
                          "acceptance": c["acceptance"]["tol_eq"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
