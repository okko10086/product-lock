# product-lock

[English](README.md) · [简体中文](README.zh-CN.md)

**A control layer that sits between a product photo and an image/video generator.**

Give it one white-background product photo. It builds a part-level fingerprint of the product,
derives a generation parameter card from that fingerprint, plans a shot list, and — after
that plan is confirmed — calls the generator, then verifies what came back.

The point is narrow and honest: **generators redraw products and lose them.** product-lock
measures that drift per part and, where it can, avoids the generator entirely by reusing the
product's own pixels.

```
product photo
   │
   ├─ build   → part-level model       (red cord / gold bead / jade part: share, Lab, texture)
   ├─ card    → generation parameter card + acceptance thresholds (same source, same card)
   ├─ plan    → shot list: camera move, first/last frame prompt, budget   (no spend)
   ├─ approve → a human confirms the plan           ← required
   ├─ run     → generator executes                  (spends credits)
   └─ verify  → per-part drift report + correction direction
```

---

## Two paths, two guarantees

| Path | Command | Product fidelity |
|---|---|---|
| **Pixel reuse** | `poster` | **Zero loss.** The product is not regenerated; it is composited from its own pixels. |
| **Generation** | `generate` / `run` | **Probabilistic.** The product *will* drift. `verify` catches how much and where. |

Do not read the second path as "product preserved". It is not. That is exactly why the first
path exists.

---

## Install

```bash
git clone <repo-url> product-lock
cd product-lock
pip install -r requirements.txt
```

Requires Python 3.9+, `opencv-python`, `numpy`, `Pillow`.

---

## Quick start

```bash
python3 core/plugin.py chain  --image examples/product.png --outdir out
python3 core/plugin.py verify --model out/model.json --image out/result.jpg
python3 core/plugin.py poster --model out/model.json --out poster.jpg \
        --title "一抹朱红" --subtitle "守一份东方雅意"
```

Generation path needs a generator CLI wired through the guard (see below):

```bash
export XYQ_CLI="$PWD/tools/xyq"          # the guard, NOT the raw CLI
python3 core/plugin.py plan    --model out/model.json --script "product hero shot" --budget 200 --out plan.json
python3 core/plugin.py approve --plan plan.json --note "shot 2 shortened by 1s"
python3 core/plugin.py run     --plan plan.json --outdir videos --confirm
```

---

## The gate: the plugin always comes first

A rule ("never generate without going through the plugin") is worthless if someone can call
the generator CLI directly. So the guard is enforced at the tool boundary, in three layers:

| Layer | Blocks | Enforced by |
|---|---|---|
| 1 | `run` without `--confirm` | plugin |
| 2 | `run --confirm` on a plan that no human approved | plugin |
| 3 | **any direct call to a credit-spending CLI command** | `tools/xyq` |

`tools/xyq` wraps the real CLI. These commands require a **single-use token**:

```
generate-video  generate-image  erase-video-subtitle
video-super-resolution  short-drama  marketing  canvas
```

A token can only be issued by `run`, only for an approved plan. It is consumed once and
expires in 15 minutes. If `XYQ_CLI` points at the raw CLI instead of the guard, the plugin
refuses to run — otherwise layer 3 is decorative.

```bash
$ tools/xyq generate-video --model ... --image ...
{"ok": false, "error": "blocked: the plugin must come first", "reason": "no token"}
# exit code 9
```

Every submit, refusal and result is appended to `logs/gate.jsonl`.

---

## Commands

| Command | Does | Input → Output |
|---|---|---|
| `build` | product photo → part-level model | image → `model.json` |
| `card` | model → parameter card | model → `card.json` |
| `verify` | per-part drift + correction direction | model + image → report |
| `generate` | call generator → download → batch verify | model → report + best image |
| `poster` | product pixels reused → poster, zero damage | model → poster |
| `plan` | script → shot list with camera moves and budget | model + script → plan |
| `run` | execute an approved plan | plan → submissions |
| `approve` | human confirmation (the gate switch) | plan → `approved: true` |
| `gate` | gate status and audit trail | — → counts + entries |
| `chain` | `build` + `card` in one go | image → model + card |
| `pose` | query the 3D pose / camera-angle library | — → angle table |
| `lexicon` | query the shot vocabulary | — → entries |

---

## Shot vocabulary

`lexicon/lexicon.json` holds the shot vocabulary (camera moves, shot language, epic moves,
story shots). It is **external and replaceable** — drop in your own film-term table, the
plugin picks it up at startup; if the file is missing, the built-in defaults are used.

```bash
python3 core/plugin.py lexicon
python3 core/plugin.py lexicon --category 运镜
```

See `docs/lexicon-format.md` for the field layout.

> The bundled file is a starter set. **Check its provenance before publishing or redistributing
> it** — shot terminology itself is generic, but a specific compilation may not be yours to ship.

---

## Camera-angle library

`pose/poses.json` and `pose/cameras.json` hold camera positions as data: azimuth,
elevation, focal length, framing. They are plain JSON — edit or replace them.

```bash
python3 core/plugin.py pose           # 9 base poses, with fitted camera distance
python3 core/plugin.py pose --cams    # 12 camera setups, with focal length
python3 tools/build_pose.py           # regenerate the tables
```

Use the table to drive Blender (or any renderer) so that a "45° view" means the same thing
every time instead of being re-invented per prompt.

---

## Known limitations

These are real, measured, and not going away soon:

- **Generation does not preserve the product.** Measured on a 5-second clip (121 frames):
  first-frame drift `0.1978`, mean `0.2315`, systematic color shift `0.31–0.37`. The problem is
  that the model does not lock onto the product — not that the camera moved.
- **Acceptance thresholds are uncalibrated.** `TOL_EQ=12 / TOL_OK=20` are provisional.
- **Part completeness is not checked.** A reference with two pendants can come back with one,
  and the current verifier will not flag it.
- **No 3D reconstruction.** Arbitrary angles need an external 3D pipeline or real multi-angle
  shots. The pixel-reuse path is limited to the angle you already photographed.
- **The trained drift model is disabled.** `driftnet_v0` collapsed (outputs converge to the
  mean); the rule layer is the default. `--use-model` opts in, and it is not trustworthy.

---

## License

MIT — see [LICENSE](LICENSE).
