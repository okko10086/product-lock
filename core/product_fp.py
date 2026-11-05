#!/usr/bin/env python3
"""产品指纹与漂移检测器 v0

输入白底商品图，输出可比的材质/纹理/颜色指纹，并用合成扰动自测检出能力。
不生成、不改图，只做"产品变没变"的量化判定。
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 自测用样本目录：默认取环境变量，没设就取当前目录下的 samples/
SRC = Path(os.environ.get("PRODUCT_FP_SRC") or (Path.cwd() / "samples"))


def imread_u(p):
    """读图并合成白底。

    修复：webp/png 的透明通道在 IMREAD_COLOR 下会变成纯黑，
    导致白底分割把整张透明背景当成产品（实测主色读出 [0,0,0.5] 就是这么来的）。
    """
    raw = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = (img[..., 3:4].astype(np.float32) / 255.0)
        rgb = img[..., :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        img = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def product_mask(bgr):
    """白底分割：从四边 floodfill 找背景，取非背景的最大连通域。

    修复：不再依赖"纯白"阈值。生成图背景常是浅灰，原逻辑会把背景当前景。
    """
    h, w = bgr.shape[:2]
    ff = bgr.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    lo = up = (14, 14, 14)
    for sx, sy in seeds:
        if m[sy + 1, sx + 1] == 0:
            cv2.floodFill(ff, m, (sx, sy), 255, loDiff=lo, upDiff=up, flags=4)
    bg = m[1:-1, 1:-1]
    fg = (1 - bg).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return fg
    # 保留所有占比 >2% 的连通域（手串是细长多段，不能只取最大块）
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a > 0.02 * h * w]
    if not keep:
        return fg
    out = np.zeros_like(fg)
    for i in keep:
        out[lab == i] = 1
    return out


def fingerprint(bgr, mask):
    m = mask.astype(bool)
    if m.sum() < 200:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0][m] / 255.0 * 100.0
    a = lab[..., 1][m] - 128.0
    b = lab[..., 2][m] - 128.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex = float(lap[m].var())
    edges = cv2.Canny(gray, 50, 150)
    edge_den = float((edges[m] > 0).mean())

    # 纹理谱：按 8 个方向的梯度能量分布（粗粒度纹理方向性）
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    hist = []
    for i in range(8):
        lo, hi = i * 22.5, (i + 1) * 22.5
        sel = m & (ang >= lo) & (ang < hi)
        hist.append(float(mag[sel].mean()) if sel.sum() > 0 else 0.0)
    tot = sum(hist) or 1.0
    hist = [round(h / tot, 4) for h in hist]

    # 主色 k-means
    px = bgr[m].reshape(-1, 3).astype(np.float32)
    K = min(3, max(1, px.shape[0] // 50))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(px, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(labels.ravel(), minlength=K)
    order = np.argsort(-cnt)
    palette = [
        {"bgr": [round(float(v), 1) for v in centers[i]], "share": round(float(cnt[i] / cnt.sum()), 3)}
        for i in order
    ]

    return {
        "area_ratio": round(float(m.mean()), 4),
        "L_mean": round(float(L.mean()), 2),
        "L_std": round(float(L.std()), 2),
        "a_mean": round(float(a.mean()), 2),
        "b_mean": round(float(b.mean()), 2),
        "a_std": round(float(a.std()), 2),
        "b_std": round(float(b.std()), 2),
        "texture_energy": round(tex, 2),
        "edge_density": round(edge_den, 4),
        "orient_hist": hist,
        "palette": palette,
    }


def norm_diff(f1, f2):
    """归一化差异：颜色/纹理/结构三块分别给分，0=完全一致。"""
    def rel(k):
        a, b = f1[k], f2[k]
        d = abs(a - b)
        return d / (abs(a) + abs(b) + 1e-6) * 2.0

    d_col = (rel("a_mean") + rel("b_mean") + rel("L_mean")) / 3.0
    d_tex = (rel("texture_energy") + rel("edge_density")) / 2.0
    d_orient = float(np.mean([abs(x - y) for x, y in zip(f1["orient_hist"], f2["orient_hist"])]))
    return {
        "color": round(d_col, 4),
        "texture": round(d_tex, 4),
        "orientation": round(d_orient, 4),
        "total": round(0.5 * d_col + 0.3 * d_tex + 0.2 * d_orient, 4),
    }


def perturb(bgr, kind):
    """合成漂移：模拟生成模型改掉产品材质/纹理/颜色。"""
    if kind == "color_shift":          # 整体色偏，模拟光照/风格串色
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 1] = np.clip(lab[..., 1] + 12, 0, 255)
        lab[..., 2] = np.clip(lab[..., 2] + 10, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    if kind == "blur":                 # 高频细节丢失，模拟"模糊"
        return cv2.GaussianBlur(bgr, (0, 0), 1.6)
    if kind == "noise":                # 噪点，模拟"生成噪"
        noise = np.random.default_rng(0).normal(0, 9, bgr.shape).astype(np.int16)
        return np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if kind == "texture_swap":         # 局部纹理替换，模拟材质变样
        h, w = bgr.shape[:2]
        out = bgr.copy()
        y0, y1 = int(h * 0.25), int(h * 0.65)
        x0, x1 = int(w * 0.25), int(w * 0.75)
        patch = out[y0:y1, x0:x1]
        k = np.ones((7, 7), np.uint8)
        out[y0:y1, x0:x1] = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, k)
        return out
    return bgr


def main():
    files = sorted([p for p in SRC.glob("*.png")])
    print(f"产品图 {len(files)} 张：", [p.name for p in files], flush=True)
    report = {}
    fps = {}
    for p in files:
        bgr = imread_u(p)
        if bgr is None:
            print("读不出", p.name)
            continue
        mask = product_mask(bgr)
        fp = fingerprint(bgr, mask)
        if fp is None:
            print("分割失败", p.name)
            continue
        fps[p.name] = (bgr, mask, fp)
        report[p.name] = {"size": list(bgr.shape[:2]), "fp": fp}
        print(f"{p.name}: 产品占比={fp['area_ratio']} L={fp['L_mean']} a={fp['a_mean']} b={fp['b_mean']} 纹理={fp['texture_energy']} 边缘={fp['edge_density']}", flush=True)

    # 自测：同一张图 × 四类扰动
    print("\n=== 漂移检出自测（正样本=轻微扰动，总分越低越像）===", flush=True)
    det = {}
    for name, (bgr, mask, fp) in fps.items():
        det[name] = {}
        for kind in ("color_shift", "blur", "noise", "texture_swap"):
            pert = perturb(bgr, kind)
            pmask = product_mask(pert)
            pfp = fingerprint(pert, pmask)
            d = norm_diff(fp, pfp) if pfp else {"total": 1.0}
            det[name][kind] = d
        print(f"{name}: " + " | ".join(f"{k}={v['total']}" for k, v in det[name].items()), flush=True)

    out = Path(os.environ.get("PRODUCT_FP_OUT") or (Path.cwd() / "产品指纹报告.json"))
    out.write_text(json.dumps({"fingerprints": report, "perturbation": det}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入", out, flush=True)


if __name__ == "__main__":
    main()
