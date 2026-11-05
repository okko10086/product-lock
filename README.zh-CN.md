# product-lock

[English](README.md) · [简体中文](README.zh-CN.md)

**夹在产品照片和生成模型之间的控制层。**

给一张白底产品图，它建出部件级指纹、推出生成参数卡、排好分镜方案，**人确认之后**才调用
生成模型，回来再逐部件验收。

它解决的问题很窄，也很实在：**生成模型会重画产品，产品就保不住了。** product-lock 逐部件
量出这个漂移，并且只要有机会，就干脆不经过生成模型——直接复用产品自己的像素。

```
产品照片
   │
   ├─ build   → 部件级模型（红绳 / 金珠 / 玉件：占比、Lab、纹理）
   ├─ card    → 参数卡 + 验收阈值（同源同卡）
   ├─ plan    → 分镜：运镜、首尾帧提示词、预算（不花钱）
   ├─ approve → 人确认方案                      ← 必须
   ├─ run     → 生成模型执行                    （花积分）
   └─ verify  → 逐部件漂移报告 + 修正方向
```

---

## 两条路，两种保证

| 路径 | 命令 | 产品保真度 |
|---|---|---|
| **像素直用** | `poster` | **零损伤**。产品不被重画，用自己的像素合成。 |
| **生成** | `generate` / `run` | **概率保证**。产品一定会偏，`verify` 量出偏多少、偏在哪。 |

第二条路不要读成"产品保住了"。它保不住。像素直用这条路就是为了这件事存在的。

---

## 安装

```bash
git clone <repo-url> product-lock
cd product-lock
pip install -r requirements.txt
```

需要 Python 3.9+、`opencv-python`、`numpy`、`Pillow`。

---

## 快速开始

```bash
python3 core/plugin.py chain  --image examples/product.png --outdir out
python3 core/plugin.py verify --model out/model.json --image out/result.jpg
python3 core/plugin.py poster --model out/model.json --out poster.jpg \
        --title "一抹朱红" --subtitle "守一份东方雅意"
```

product-lock 不绑定任何生成服务，写一个 CLI 接进来就行，约定见 [docs/adapter.md](docs/adapter.md)。
先用自带的假生成器把链路跑通，不花一分钱：

```bash
export GENERATOR_CLI_REAL="$PWD/examples/mock_generator.sh"
python3 core/plugin.py plan    --model out/model.json --script "产品主视觉" --budget 200 --out plan.json
python3 core/plugin.py approve --plan plan.json --note "第 2 镜改短 1 秒"
python3 core/plugin.py run     --plan plan.json --outdir videos --confirm
```

---

## 闸门：插件永远在前

"不许绕过插件出片"这条规矩，如果别人直接调 CLI 就能出，那它是废话。
所以闸门修在**工具边界**上，三层：

| 层 | 拦什么 | 谁把关 |
|---|---|---|
| 1 | `run` 不带 `--confirm` | 插件 |
| 2 | `run --confirm` 但方案没人确认过 | 插件 |
| 3 | **任何直接调花积分动词的调用** | `tools/guard` |

`tools/guard` 把你的 CLI 包在里面。这些动词必须先拿到一枚**一次性令牌**：

```
video   image        # 可用 GUARD_SPEND_VERBS 覆盖
```

令牌只能由 `run` 为已确认的方案签发，用一次作废，15 分钟过期。
`GENERATOR_CLI` 如果指向真 CLI 而不是守门人，插件直接拒绝执行——否则第三层就是摆设。

```bash
$ tools/guard video --image ... --prompt ...
{"ok": false, "error": "blocked: the plugin must come first", "reason": "no token"}
# 退出码 9
```

每次提交、拒绝、结果都追加到 `logs/gate.jsonl`。

---

## 命令

| 命令 | 作用 | 输入 → 输出 |
|---|---|---|
| `build` | 产品图 → 部件级模型 | 图 → `model.json` |
| `card` | 模型 → 参数卡 | model → `card.json` |
| `verify` | 逐部件漂移 + 修正方向 | model + 图 → 报告 |
| `generate` | 调生成模型 → 下载 → 批量验收 | model → 报告 + 最佳图 |
| `poster` | 产品像素直用 → 海报，零损伤 | model → 海报 |
| `plan` | 剧本 → 分镜（运镜 + 预算） | model + 剧本 → 方案 |
| `run` | 执行已确认方案 | 方案 → 提交记录 |
| `approve` | 人工确认（闸门开关） | 方案 → `approved: true` |
| `gate` | 闸门状态与流水 | — → 计数 + 明细 |
| `chain` | 一次跑完 `build` + `card` | 图 → model + card |
| `pose` | 查 3D 姿态 / 机位库 | — → 机位表 |
| `lexicon` | 查词库 | — → 条目 |

---

## 词库

`lexicon/lexicon.json` 放词库（运镜、镜头语言、史诗运镜、剧情镜头）。它是**外置且可整体
替换**的：放自己的影视术语表进去，启动时自动覆盖内置默认；文件不在就用内置的。

```bash
python3 core/plugin.py lexicon
python3 core/plugin.py lexicon --category 运镜
```

字段口径见 `docs/lexicon-format.md`。

> 仓库自带的是起步集。**发布或再分发前请自行确认它的来源与授权**——影视术语本身是通用的，
> 但某一份具体的整理稿未必有权利随代码一起发。

---

## 机位库

`pose/poses.json` 和 `pose/cameras.json` 把机位当数据存着：方位角、俯仰角、焦段、景别。
纯 JSON，直接改或整份替换都行。

```bash
python3 core/plugin.py pose           # 9 个基础机位，带拟合出的相机距离
python3 core/plugin.py pose --cams    # 12 个机位，带焦段
python3 tools/build_pose.py           # 重新生成这两张表
```

用这张表驱动 Blender（或任何渲染器），让"45° 视角"每次都指同一个东西，而不是每个提示词
重新发明一遍。

---

## 已知边界

这些是真测出来的，不是保留意见：

- **生成保不住产品。** 5 秒 121 帧实测：首帧漂移 `0.1978`、均值 `0.2315`、系统性偏色
  `0.31–0.37`。问题在模型没锁定产品，不在运镜。
- **验收阈值没标定。** `TOL_EQ=12 / TOL_OK=20` 是暂定值。
- **部件完整性没检测。** 参考图两个玉件、生成图只剩一个，现在的验收器查不出来。
- **不做 3D 重建。** 任意角度需要外部 3D 流程或多角度实拍；像素直用只能在已经拍到的角度用。
- **训练出的漂移模型是关掉的。** `driftnet_v0` 塌缩（输出趋均值），默认走规则层；
  `--use-model` 才启用，且不可信。

---

## 许可证

MIT — 见 [LICENSE](LICENSE)。
