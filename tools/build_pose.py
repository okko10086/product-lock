"""生成仓库自带的机位库。

机位配置（方位角 / 俯仰角 / 焦段 / 景别）是摄影常识的组合，说明文字为本项目撰写。
"""
import json
from pathlib import Path

POSES = {
    "note": "基础机位表。azimuth 绕产品竖轴，0=正面；elevation 相机高度角，正值俯视；"
            "dist 为拟合出的相机距离（场景单位），fill 为产品占画幅比例。",
    "poses": [
        {"id": "P01", "name": "front_eye", "label": "正面平视", "azimuth": 0, "elevation": 5,
         "dist": 9.71, "fill": 0.82, "use": "主图、开场定场"},
        {"id": "P02", "name": "front_45", "label": "正面 45 度", "azimuth": 45, "elevation": 12,
         "dist": 8.01, "fill": 0.82, "use": "最常见的产品角"},
        {"id": "P03", "name": "side_90", "label": "侧面 90 度", "azimuth": 90, "elevation": 8,
         "dist": 4.15, "fill": 0.82, "use": "交代厚度与侧面结构"},
        {"id": "P04", "name": "front_left_45", "label": "左前 45 度", "azimuth": -45, "elevation": 12,
         "dist": 8.01, "fill": 0.82, "use": "反向补角"},
        {"id": "P05", "name": "top_down", "label": "顶视", "azimuth": 0, "elevation": 75,
         "dist": 9.44, "fill": 0.82, "use": "俯视布局"},
        {"id": "P06", "name": "ground_level", "label": "贴地视角", "azimuth": 20, "elevation": 1.5,
         "dist": 9.30, "fill": 0.82, "use": "平齐产品高度看过去"},
        {"id": "P07", "name": "back", "label": "背面", "azimuth": 180, "elevation": 8,
         "dist": 9.76, "fill": 0.82, "use": "反打，交代背面"},
        {"id": "P08", "name": "macro", "label": "微距特写", "azimuth": 30, "elevation": 10,
         "dist": 2.05, "fill": 0.92, "use": "材质细节"},
        {"id": "P09", "name": "high_45", "label": "斜俯 45 度", "azimuth": 135, "elevation": 40,
         "dist": 7.94, "fill": 0.82, "use": "斜俯全览"},
    ],
}

CAMERAS = {
    "note": "机位库：焦段 + 机位角度 + 景别。focal 单位为 mm（全画幅等效）；"
            "framing 里 full=整组完整入画，mid=主体群的一半，close=单件，detail=单件局部。",
    "focal_note": "28mm 环境多、35mm 略广、50mm 接近人眼、85mm 压缩背景、100mm 特写不畸变。",
    "framing_note": "景别决定框住多少，焦段决定透视压缩多少，两者要一起定。",
    "shots": [
        {"id": "C01", "name": "front_eye", "label": "正面平视", "focal": 50,
         "azimuth": 0, "elevation": 5, "framing": "full",
         "desc": "机位与主体同高，整组完整入画"},
        {"id": "C02", "name": "front_high", "label": "正面俯视", "focal": 50,
         "azimuth": 0, "elevation": 45, "framing": "mid",
         "desc": "机位抬高向下看，只框主体群中间一半"},
        {"id": "C03", "name": "front_low", "label": "正面仰视", "focal": 35,
         "azimuth": 0, "elevation": -30, "framing": "full",
         "desc": "机位低于主体向上看；台面会挡住视线，需悬空拍摄"},
        {"id": "C04", "name": "side_eye", "label": "侧面平视", "focal": 50,
         "azimuth": 90, "elevation": 5, "framing": "full",
         "desc": "从正侧面看，交代厚度"},
        {"id": "C05", "name": "front_45_close", "label": "前侧 45 度", "focal": 50,
         "azimuth": 45, "elevation": 5, "framing": "close",
         "desc": "前侧 45 度平视，只框单件"},
        {"id": "C06", "name": "rear_45_close", "label": "侧后 45 度", "focal": 85,
         "azimuth": 135, "elevation": 10, "framing": "close",
         "desc": "从侧后方拍，长焦压缩背景"},
        {"id": "C07", "name": "over_shoulder", "label": "过肩", "focal": 85,
         "azimuth": 30, "elevation": 8, "framing": "close",
         "desc": "前景物件虚化成一条边，主体在后"},
        {"id": "C08", "name": "top_down", "label": "正上方顶拍", "focal": 35,
         "azimuth": 0, "elevation": 90, "framing": "full",
         "desc": "垂直向下，画面变成平面布局"},
        {"id": "C09", "name": "wide_env", "label": "广角远景", "focal": 28,
         "azimuth": 0, "elevation": 5, "framing": "wide",
         "desc": "远距离平视，主体占比很小，保留完整环境"},
        {"id": "C10", "name": "full_shot", "label": "全景", "focal": 50,
         "azimuth": 30, "elevation": 5, "framing": "full",
         "desc": "前侧平视，主体完整入画，上下留少量空间"},
        {"id": "C11", "name": "medium_close", "label": "中近景", "focal": 85,
         "azimuth": 0, "elevation": 5, "framing": "close",
         "desc": "眼睛高度、长焦，框住单件"},
        {"id": "C12", "name": "detail", "label": "细节特写", "focal": 100,
         "azimuth": 30, "elevation": 12, "framing": "detail",
         "desc": "前侧轻微俯拍，只留局部材质"},
    ],
    "framing_scale": {
        "wide": "主体很小，先交代它在哪",
        "full": "主体完整入画",
        "mid": "只框主体群的一半，看关系",
        "close": "框单件",
        "detail": "只留局部，看材质",
    },
}

out = Path(__file__).resolve().parent.parent / "pose"
out.mkdir(parents=True, exist_ok=True)
(out / "poses.json").write_text(json.dumps(POSES, ensure_ascii=False, indent=1), encoding="utf-8")
(out / "cameras.json").write_text(json.dumps(CAMERAS, ensure_ascii=False, indent=1), encoding="utf-8")
print("写入", out)
print("姿态", len(POSES["poses"]), "个 | 机位", len(CAMERAS["shots"]), "个")
