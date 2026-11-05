#!/usr/bin/env python3
"""产品一致性控制插件 v1

三层链路（对应"大模型控制小模型去控制视觉模型"）：
  大模型  —— 调用本插件，给目标描述，读验收结论（不碰参数细节）
  小模型  —— 本插件内的 DriftNet，负责建模/预测漂移/给修正方向
  视觉模型 —— 按参数卡出图（Seedream / 小云雀），出完由本插件验收

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




# ---------------- generate / poster：编排视觉模型 ----------------

def _xyq_call(card: dict, count: int = 2, model: str = "Seedream 5.0 Flash",
              ratio: str = "6", resolution: str = "2K"):
    """调用小云雀生图。返回提交信息；调用失败抛异常，不吞错。"""
    import os, subprocess, json as _json
    cli = _guard_cli()
    cmd = [cli, "generate-image", "--prompt", card["prompt"], "--model", model,
           "--image", card["source_model"], "--ratio", ratio, "--resolution", resolution,
           "--generate-image-count", str(count), "--source", "dsh"]
    env = dict(os.environ)
    env["DSH_PLUGIN_TOKEN"] = gate.issue_token_for_image(card, model, count)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        raise SystemExit(f"提交失败: {(r.stderr or r.stdout)[-400:]}")
    return _json.loads((r.stdout or "").strip().splitlines()[-1])


def _xyq_fetch(thread: str, run: str, outdir: str, cli: str, timeout_s: int = 600):
    """轮询并下载结果。"""
    import subprocess, time
    Path(outdir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = subprocess.run([cli, "query-result", "--thread-id", thread, "--run-id", run,
                            "--download-dir", outdir], capture_output=True, text=True)
        out = r.stdout or ""
        if '"completed": true' in out:
            return sorted(str(x) for x in Path(outdir).glob("*.jpeg"))
        time.sleep(10)
    raise SystemExit(f"轮询超时（{timeout_s}s）")


GUARD = Path(__file__).resolve().parent.parent / "tools" / "xyq"


def _guard_cli() -> str:
    """拿 CLI 路径，并确认它指向守门人；指向真 CLI 就直接拒。

    插件内部拦得住自己，拦不住直接调 CLI 出片，所以生成入口必须是守门人。
    """
    import os
    cli = os.environ.get("XYQ_CLI") or str(GUARD)
    if Path(cli).name != "xyq":
        gate.audit("exec_refuse", reason="XYQ_CLI 未指向守门人", cli=cli)
        raise SystemExit(
            "闸门拒绝：XYQ_CLI 指向的是真 CLI，能绕过插件直接出片。\n"
            f"改成守门人再用：export XYQ_CLI={GUARD}"
        )
    return cli


def cmd_generate(model_path: str, count: int, outdir: str):
    """card -> 视觉模型 -> 下载结果 -> 逐个验收。"""
    m = json.loads(Path(model_path).read_text(encoding="utf-8"))
    card = make_card(m)
    gate.log_submit("image", model=model_path, count=count, prompt=card.get("prompt", "")[:120])
    info = _xyq_call(card, count=count)
    gate.log_result("image", True, thread_id=info.get("thread_id"), run_id=info.get("run_id"), count=count)
    import os
    import os as _os
    files = _xyq_fetch(info["thread_id"], info["run_id"], outdir, _os.environ.get("XYQ_CLI") or str(GUARD))
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

# 运镜词库（团队内部整理）
# safe=True 表示产品不变形；False 进禁用表，方案里不会出现
# 词库 · 运镜（24 种）
# safe=True 不改产品几何；False 需重绘透视或换视角，产品会变形
CAMERA_LIB = {
    # ---- 一、相机运动 ----
    "横摇": {"en": "Pan", "safe": True, "how": "相机原地向左或向右转动",
             "param": "小角度 ≤15°，速度 ≤20 px/s", "risk": "角度大会带出画"},
    "俯仰": {"en": "Tilt", "safe": True, "how": "相机原地向上或向下摇动",
             "param": "小角度 ≤10°", "risk": "大角度改变透视"},
    "推近": {"en": "Dolly in", "safe": True, "how": "相机向主体靠近",
             "param": "3-5s 推 8-12%，≤20 px/s", "risk": "推太多边缘出画"},
    "变焦推近": {"en": "Zoom in", "safe": True, "how": "机位不动，用变焦放大主体",
                "param": "1.2-1.8s 走完", "risk": "几乎无，透视不变"},
    "跟拍": {"en": "Tracking shot", "safe": False, "how": "相机与移动主体同步行进",
             "param": "—", "risk": "产品需整体移动，单图无法稳定跟随"},
    "环绕": {"en": "Orbit (Arc shot)", "safe": False, "how": "相机绕主体做弧线运动",
             "param": "—", "risk": "需多角度素材，单视角硬转=贴片挤扁"},
    "摇臂上升": {"en": "Crane up", "safe": False, "how": "相机向上升高，展开空间",
              "param": "小角度 ≤5% 可放行", "risk": "大角度改变透视，产品变形"},
    "推轨变焦": {"en": "Dolly zoom (Vertigo)", "safe": False, "how": "主体大小近似不变，背景透视变化",
              "param": "—", "risk": "背景重绘，图生视频必变形"},

    # ---- 二、焦点与时间 ----
    "拉焦": {"en": "Rack focus", "safe": True, "how": "在前景与后景之间切换焦点",
             "param": "1.2-1.8s 走完", "risk": "无，形状不变"},
    "手持": {"en": "Handheld", "safe": True, "how": "用细微晃动营造现场感",
             "param": "幅度 ≤2% 画面", "risk": "晃动大会让产品边缘抖"},
    "第一人称穿越": {"en": "FPV drone shot", "safe": False, "how": "模拟穿越机高速飞行的视角",
                "param": "—", "risk": "透视大改，产品跟着改形"},
    "荷兰角": {"en": "Dutch angle", "safe": True, "how": "倾斜画面，制造不安与张力",
              "param": "≤8°", "risk": "画面歪，但产品形状不变"},
    "慢动作": {"en": "Slow motion", "safe": True, "how": "放慢动作，突出过程与细节",
              "param": "0.25x-0.5x", "risk": "无"},
    "速度渐变": {"en": "Speed ramp", "safe": True, "how": "在正常、快放与慢放间变速",
               "param": "单段变速 ≤2 次", "risk": "无"},
    "延时摄影": {"en": "Time-lapse", "safe": True, "how": "加快时间，呈现缓慢变化",
              "param": "—", "risk": "无"},
    "定格": {"en": "Freeze frame", "safe": True, "how": "暂时停住画面，强调瞬间",
             "param": "0.5-1s", "risk": "无"},

    # ---- 三、剪辑与转场（不改画面几何）----
    "跳切": {"en": "Jump cut", "safe": True, "how": "相似机位中跳过一段时间",
             "param": "—", "risk": "无"},
    "匹配剪辑": {"en": "Match cut", "safe": True, "how": "利用相似形状或动作衔接场景",
              "param": "—", "risk": "无"},
    "甩镜转场": {"en": "Whip pan", "safe": False, "how": "快速横摇，借运动模糊衔接",
              "param": "2-3 帧", "risk": "动作段速度超稳态线，产品易糊"},
    "叠化": {"en": "Cross dissolve", "safe": True, "how": "前后画面重叠并逐渐过渡",
             "param": "0.5-1s", "risk": "无"},
    "划像转场": {"en": "Wipe", "safe": True, "how": "边界扫过，逐步换成下一画面",
              "param": "0.5-1s", "risk": "无"},
    "突变剪辑": {"en": "Smash cut", "safe": True, "how": "突然切换，制造强烈反差",
              "param": "—", "risk": "无"},
    "声音先入": {"en": "J-cut", "safe": True, "how": "下一场的声音先于画面出现",
              "param": "—", "risk": "无（音频层）"},
    "声音延续": {"en": "L-cut", "safe": True, "how": "上一场的声音延续到下一画面",
              "param": "—", "risk": "无（音频层）"},
}

# 提示词公式（词库原话）：镜头名称 + 时长 + 速度 + 主体/方向
CAMERA_PROMPT_FORMULA = "镜头名称 + 时长 + 速度 + 主体/方向"
CAMERA_PROMPT_EXAMPLE = "3秒 Dolly in，缓慢推近人物面部，保持视线水平。"


# ========== 词库完整运镜库（第二版，来源逐条标注）==========

# 词库 · 镜头语言 1/4 基础运动
SHOT_LANG_BASIC = {
    "横摇":       {"en": "Pan",        "how": "原地向左右转动"},
    "俯仰":       {"en": "Tilt",       "how": "原地向上下抬落"},
    "推近":       {"en": "Dolly in",   "how": "镜头向前靠近"},
    "拉远":       {"en": "Dolly out",  "how": "镜头向后退开"},
    "横移":       {"en": "Truck",      "how": "镜头向侧面平移"},
    "升降":       {"en": "Pedestal",   "how": "朝向不变，垂直升降"},
    "摇臂":       {"en": "Crane",      "how": "借摇臂升高并向下看"},
    "环绕":       {"en": "Orbit (Arc)","how": "绕主体画弧线移动"},
    "滚转":       {"en": "Roll",       "how": "绕镜头轴心旋转"},
}

# 词库 · 镜头语言 2/4 特殊运动
SHOT_LANG_SPECIAL = {
    "变焦推近":   {"en": "Zoom in",    "how": "机位不变，增大焦距"},
    "推轨变焦":   {"en": "Dolly zoom", "how": "后移并变焦推近，主体大小不变"},
    "跟拍":       {"en": "Tracking",   "how": "从侧面跟随移动"},
    "前引跟拍":   {"en": "Leading",    "how": "镜头在前方引导"},
    "后跟拍":     {"en": "Follow",     "how": "从后方跟随"},
    "甩镜":       {"en": "Whip pan",   "how": "瞬间快速横摇"},
    "手持":       {"en": "Handheld",   "how": "手持拍摄的轻微晃动"},
    "无人机":     {"en": "Drone",      "how": "升空并俯拍"},
    "主观视角":   {"en": "POV",        "how": "以主体视角前进"},
}

# 词库 · 史诗运镜（5 种）
EPIC_MOVES = {
    "揭示升镜":   {"how": "镜头从主体升起，环境全貌压过来，一个动作给出敬畏", "safe": True},
    "庄严缓推":   {"how": "极慢推向远处的主体，旅程读作命运", "safe": True},
    "横移建立全景": {"how": "横向漂移，前中远景不同速，视差卖出尺度", "safe": True},
    "低角度纪念碑式": {"how": "低角度仰视，让主体有历史分量", "safe": True},
    "静守苍茫":   {"how": "锁定极远景，主体只占画面下方一小块", "safe": True},
}

# 词库 · 剧情镜头（20 组，每条含完整提示词）
STORY_SHOTS = [
    {"no": 1, "name": "背影跟进", "use": "夜巷入场 / 身份悬念",
     "prompt": "6秒，单一连续镜头。镜头在青灰衣剑客身后约两米，以腰部高度平稳跟进。他走过雨后窄巷，始终位于画面中部；结尾停在亮着油灯的门前，轻侧头但不转身。巷道透视连贯，披风随步伐摆动。"},
    {"no": 2, "name": "门框揭示", "use": "藏身现身 / 突然登场",
     "prompt": "6秒，单一连续镜头。近处木门框遮住画面右侧，镜头缓慢向右平移，逐渐露出门后的绛红衣女子。她先垂眼，再抬头看向走廊尽头。门框随视差移出画面，人物站位不变，最后停在中景。"},
    {"no": 3, "name": "低位抬升", "use": "强者登场 / 气场变化",
     "prompt": "6秒，单一连续镜头。镜头从剑客靴边的低机位缓慢升至胸口高度，同时轻微向上仰拍。他站在雪山石阶顶端，衣摆被风吹动，始终不向镜头走近。结尾呈现上半身与身后的山脊，动作稳定。"},
    {"no": 4, "name": "环境推进", "use": "陌生地界 / 主角入局",
     "prompt": "6秒，单一连续镜头。从竹海中的石桥远景开始，镜头沿桥面轴线匀速向前推进，逐渐接近桥中央的绛红衣女子。她握着折扇望向雾里，结尾停在全身中景。桥栏、竹林和人物的空间关系保持一致。"},
    {"no": 5, "name": "侧向并行", "use": "追逐 / 轻功奔跑",
     "prompt": "6秒，单一连续镜头。镜头与青灰衣剑客保持侧向距离，沿竹林小路同步平移。他从左向右持续奔跑，始终留在画面中部；前景竹干快速掠过，远景移动较慢。结尾放慢脚步，移动方向不反转。"},
    {"no": 6, "name": "贴地推近", "use": "拔剑起手 / 攻势预备",
     "prompt": "6秒，单一连续镜头。镜头贴近石台地面缓慢前进，低角度拍摄剑客拔剑。先看见靴子与剑鞘，再随拔剑动作轻抬镜头，停在半身构图。剑刃与手的位置稳定，人物不扑向镜头，不出现血腥画面。"},
    {"no": 7, "name": "前景带速度", "use": "交锋 / 擦身而过",
     "prompt": "6秒，单一连续镜头。镜头固定在庭院侧方，青灰衣剑客与绛红衣女子从左右接近，格挡后错身分开。一柄剑短暂掠过近处前景，带出运动模糊，面部和另一人物仍清楚。两人不换位瞬移，结尾各自停步。"},
    {"no": 8, "name": "退镜揭局", "use": "包围 / 攻守反转",
     "prompt": "6秒，单一连续镜头。从剑客持剑的中景开始，镜头平稳后退，逐渐露出两侧廊柱后的对手。剑客原地缓缓转头，察觉包围；结尾停在能看清三人站位的全景。退镜不穿墙，人物数量与位置连续。"},
]



# 词库 · 镜头语言 3/4 景别与角度
SHOT_LANG_FRAMING = {
    "大远景":   {"en": "Extreme long shot", "how": "环境很大，主体很小"},
    "全景":     {"en": "Full shot",         "how": "从头到脚完整入镜"},
    "中景":     {"en": "Medium shot",       "how": "取到腰部以上"},
    "特写":     {"en": "Close-up",          "how": "主体占满画面"},
    "大特写":   {"en": "Extreme close-up",  "how": "只呈现局部细节"},
    "高角度":   {"en": "High angle",        "how": "向下俯拍，主体显小"},
    "低角度":   {"en": "Low angle",         "how": "向上仰拍，突出力量感"},
    "鸟瞰":     {"en": "Bird's-eye",        "how": "正上方俯拍"},
    "荷兰角":   {"en": "Dutch angle",       "how": "画面倾斜，营造紧张感"},
}

# 词库 · 镜头语言 4/4 构图与焦点
SHOT_LANG_COMPOSITION = {
    "越肩镜头": {"en": "Over-the-shoulder", "how": "从肩膀上方越过拍摄"},
    "双人镜头": {"en": "Two shot",          "how": "两个主体同时入镜"},
    "广角镜头": {"en": "Wide lens 12mm",    "how": "近距离拍摄，透视夸张"},
    "长焦镜头": {"en": "Telephoto lens",    "how": "远处取景，背景显得压缩"},
    "浅景深":   {"en": "Shallow focus",     "how": "背景虚化"},
    "深焦":     {"en": "Deep focus",        "how": "前景与背景都清晰"},
    "拉焦":     {"en": "Rack focus",        "how": "焦点从前景切到后景"},
    "揭示镜头": {"en": "Reveal",            "how": "从遮挡物后方展示画面"},
    "三分法构图": {"en": "Rule of thirds",  "how": "主体放在三分线交点"},
}

# 三者辨析：推近 / 变焦推近 / 推轨变焦
CAMERA_DIFF = {
    "推近 Dolly in": "机位向主体靠近，透视关系改变，能看到前后景视差",
    "变焦推近 Zoom in": "机位不动，增大焦距收紧取景，透视基本不变",
    "推轨变焦 Dolly zoom": "机位前后移动同时反向变焦，主体大小不变但背景透视拉伸",
    "怎么选": "想改空间关系→推近；同机位收紧取景→变焦；要背景拉伸感→推轨变焦",
}



# 词库 · 剧情镜头（续：情绪/关系/衔接）
STORY_SHOTS_MORE = [
    {"no": 9, "name": "越肩缓推", "use": "质问 / 藏着话不说",
     "prompt": "6秒，单一连续镜头。前景保留剑客虚化的肩部，镜头缓慢推向对面的绛红衣女子。她先直视他，随后轻抿唇、垂眼，结尾停在面部近景。镜头始终位于两人连线同一侧，前景肩部逐渐退出。"},
    {"no": 10, "name": "绕到眼神", "use": "认出故人 / 情绪转折",
     "prompt": "6秒，单一连续镜头。镜头从女子背后偏侧的位置开始，缓慢绕到她侧前方约四十五度。她始终望向同一个画外的人，眼睛渐亮，嘴角轻颤。结尾停在侧近景，不绕满一圈，不突然越过两人的关系轴线。"},
    {"no": 11, "name": "停镜留反应", "use": "告别 / 强撑后的失落",
     "prompt": "6秒，固定近景，不移动镜头。剑客看着画外的人，先缓缓点头，再挤出很浅的笑。对方离开后，他的视线停留，笑容逐渐消失，下颌轻绷。只保留轻微呼吸，无对白；不要用突然落泪替代表情变化。"},
    {"no": 12, "name": "拉远留空", "use": "孤独 / 关系结束",
     "prompt": "6秒，单一连续镜头。从女子握住折扇的半身中景开始，镜头缓慢后退，逐渐露出空旷渡口。她始终站在原地，缓缓放下手；结尾人物变小，身旁留出明显空位。环境逐渐进入画面，不使用变焦跳变。"},
    {"no": 13, "name": "前后景对峙", "use": "试探 / 谁先开口",
     "prompt": "6秒，单一连续镜头。女子站在左侧近景，剑客站在右侧远景，中间隔着一段走廊。镜头缓慢侧移，两人始终看向彼此，距离不变。结尾用同一画面保留双方，让观众看清试探关系，不切成两个正脸特写。"},
    {"no": 14, "name": "一步进入同框", "use": "安慰 / 决定靠近",
     "prompt": "6秒，单一连续镜头。起始是女子站在窗边的中景，剑客从画面右侧走近一步。镜头轻微向左平移，为两人留出空间；结尾停在双人中景，他递出一件披风她抬眼回应。人物不突然贴近，手与披风完整可见。"},
    {"no": 15, "name": "侧移揭示同行", "use": "误会化解 / 并肩同行",
     "prompt": "6秒，单一连续镜头。从女子侧面的行走中景开始，镜头缓慢向外侧平移，逐渐露出与她同方向前行的剑客。两人步速一致，保持半步间距；结尾是并肩双人中景。镜头用侧移揭示关系，不让人物凭空出现。"},
    {"no": 16, "name": "小弧线靠近", "use": "低声交谈 / 关系升温",
     "prompt": "6秒，单一连续镜头。两人站在竹亭内，镜头围绕他们缓慢移动约三十度，同时轻微靠近。女子抬眼，剑客低头倾听；结尾停在双人近景，保留一臂以内的自然距离。背景有轻微视差，不旋转满圈、不突然亲吻。"},
    {"no": 17, "name": "柱影遮挡接场", "use": "室外进室内 / 自然换景",
     "prompt": "分两段生成，再剪辑衔接。前段：镜头向右跟随剑客走过廊柱，柱子完全遮满画面时结束。后段：从相同深色遮挡开始，继续向右移出，露出藏书厅里的他。两段移动方向、速度和人物服装一致，在全遮挡处切换。"},
    {"no": 18, "name": "同向甩镜接场", "use": "追逐延续 / 节奏提速",
     "prompt": "分两段生成，再剪辑衔接。前段：剑客向右奔跑，镜头向右快速横摇，以横向模糊结束。后段：从相似的右向模糊起始，镜头稳定后露出他继续奔跑。统一人物运动方向，在模糊最强处剪接；不要一次要求多个地点。"},
    {"no": 19, "name": "相同动作接场", "use": "时间跳转 / 动作不断",
     "prompt": "分两段生成，再做动作匹配剪辑。前段：女子在渡口收起折扇，扇骨合拢到一半时结束。后段：她在竹亭内，以相同手势和扇子位置继续合拢。两段机位、景别与手部方向接近，在同一动作阶段切换。"},
    {"no": 20, "name": "同景光线过渡", "use": "时间流逝 / 氛围变化",
     "prompt": "6秒，单一连续镜头。镜头固定在渡口的同一位置，人物静止望向河面，画面光线从暮色逐渐变为蓝色夜光。保持人物、船只和岸线位置稳定，只让灯笼缓缓亮起。表现时间过渡，不让建筑变形或人物消失。"},
]


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
    cli = _guard_cli()
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
        cmd = [cli, "generate-video", "--model", s["recommended_model"],
               "--image", first_img, "--prompt", s["camera_instruction"] + "，" + s["last_frame_prompt"],
               "--ratio", s["ratio"], "--duration", str(s["duration_s"]), "--source", "dsh"]
        env = dict(os.environ); env["DSH_PLUGIN_TOKEN"] = token
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            done.append({"shot": s["shot_id"], "ok": False, "error": (r.stderr or r.stdout)[-300:]})
            gate.log_result("video", False, shot=s["shot_id"], error=(r.stderr or r.stdout)[-200:])
            continue
        info = json.loads((r.stdout or "").strip().splitlines()[-1])
        done.append({"shot": s["shot_id"], "ok": True, "thread_id": info["thread_id"],
                     "run_id": info["run_id"], "credits": s["credit_estimate"]})
        gate.log_result("video", True, shot=s["shot_id"], thread_id=info["thread_id"],
                        run_id=info["run_id"], credits=s["credit_estimate"])
        time.sleep(2)
    res = {"outdir": outdir, "submitted": done,
           "total_credits": sum(x.get("credits", 0) for x in done if x.get("ok"))}
    Path(outdir, "_run.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


POSE_DEF = Path(__file__).resolve().parent.parent.parent / "姿态库" / "姿态定义.json"
CAM_DEF = Path(__file__).resolve().parent.parent.parent / "姿态库" / "机位库_12.json"


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
                "shots": [{"id": x["id"], "名": x["名"], "lens": x["lens"],
                           "az": x["azimuth"], "el": x["elevation"], "构图": x["framing"],
                           "原话": x["原话"]} for x in shots]}
    poses = load_poses()
    if not poses:
        raise SystemExit(f"读不到姿态库: {POSE_DEF}")
    if pid:
        poses = [p for p in poses if p["id"].upper() == pid.upper()]
        if not poses:
            raise SystemExit(f"没有这个姿态: {pid}")
    if blender:
        return {"poses": [{"id": p["id"], "名": p["名"], "azimuth": p["azimuth"],
                           "elevation": p["elevation"], "dist": p.get("fit_dist"),
                           "fill": p.get("fit_fill", 0.82)} for p in poses]}
    return {"count": len(poses), "source": str(POSE_DEF),
            "poses": [{"id": p["id"], "名": p["名"], "az": p["azimuth"],
                       "el": p["elevation"], "dist": p.get("fit_dist"), "用途": p.get("用途", "")}
                      for p in poses]}


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
