"""闸门：插件永远在前。

任何消耗积分的生成动作（图片/视频）都必须留下一条可追溯的记录：
  plan -> approve（人工确认，落盘） -> run
中间不允许跳过 approve，也不允许用别的入口直接调生成模型。

日志：logs/gate.jsonl，一行一次事件，只追加不改写。
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "gate.jsonl"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def audit(event: str, **fields) -> dict:
    """记一条闸门事件。永远不抛异常，记不上也不影响主流程。"""
    row = {"ts": _now(), "event": event, **fields}
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return row


def approve(plan_path: str, note: str = "", by: str = "user") -> dict:
    """人工确认一个方案：在 plan 文件里写 approved。这一步只能由人发起。"""
    p = Path(plan_path)
    plan = json.loads(p.read_text(encoding="utf-8"))
    plan["approved"] = True
    plan["approved_at"] = _now()
    plan["approved_by"] = by
    plan["approved_note"] = note
    p.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    audit("approve", plan=str(p), credits=plan.get("total_credits_estimate"),
          shots=len(plan.get("shots", [])), note=note, by=by)
    return plan


def require_approved(plan: dict, plan_path: str = "") -> None:
    """没人工确认过的方案，一律拒绝，并留痕。"""
    if not plan.get("approved"):
        audit("refuse", reason="方案未人工确认", plan=plan_path,
              credits=plan.get("total_credits_estimate"))
        raise SystemExit(
            "闸门拒绝：方案未经人工确认，不会消耗积分。\n"
            f"先让人确认，再执行：python3 core/plugin.py approve --plan {plan_path}"
        )


def log_submit(kind: str, **fields) -> None:
    audit("submit", kind=kind, **fields)


def log_result(kind: str, ok: bool, **fields) -> None:
    audit("result", kind=kind, ok=ok, **fields)


def status(limit: int = 15) -> dict:
    """计数按全量日志算，明细只取最近 limit 条。"""
    rows = []
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    counts = {"approve": 0, "refuse": 0, "submit": 0, "result": 0}
    for r in rows:
        e = r.get("event")
        if e in counts:
            counts[e] += 1
    return {
        "rule": "插件永远在前：plan -> approve -> run，任何生成动作绕不过去",
        "log": str(LOG),
        "total_events": len(rows),
        "counts": counts,
        "recent": rows[-limit:],
    }


# ---------- 令牌：把闸门修到工具边界上 ----------
# 插件内部拦得住自己，拦不住别人直接调小云雀 CLI。
# 所以真正出片前要持一枚一次性令牌，令牌只能由 run 在方案被确认后签发。

TOKENS = ROOT / "logs" / "tokens"
TTL_SECONDS = 900


def issue_token(plan_path: str, shot: int, model: str, credits: float, ttl: int = TTL_SECONDS) -> str:
    """签发一次性令牌。只有 approve 过的方案才签得出来。"""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if not plan.get("approved"):
        audit("token_refuse", reason="方案未确认，不签令牌", plan=plan_path, shot=shot)
        raise SystemExit("拒绝签发令牌：方案未人工确认")
    TOKENS.mkdir(parents=True, exist_ok=True)
    tok = f"{int(time.time())}-{shot}-{os.getpid()}-{secrets.token_hex(4)}"
    row = {"token": tok, "plan": str(Path(plan_path).resolve()), "shot": shot,
           "model": model, "credits": credits, "issued_at": time.time(),
           "expires_at": time.time() + ttl, "consumed": False}
    (TOKENS / f"{tok}.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    audit("token_issue", token=tok, plan=plan_path, shot=shot, model=model, credits=credits)
    return tok


def consume_token(tok: str) -> dict | None:
    """核销令牌。核销即失效，重复用不了；过期也不认。"""
    if not tok:
        return None
    p = TOKENS / f"{tok}.json"
    if not p.exists():
        return None
    try:
        row = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if row.get("consumed"):
        return None
    if time.time() > row.get("expires_at", 0):
        audit("token_expired", token=tok)
        return None
    row["consumed"] = True
    p.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return row


def issue_token_for_image(card: dict, model: str, count: int, ttl: int = TTL_SECONDS) -> str:
    """生图令牌。生图不要求先出分镜，但必须留痕、且只能由插件签发。"""
    TOKENS.mkdir(parents=True, exist_ok=True)
    tok = f"{int(time.time())}-img-{os.getpid()}-{secrets.token_hex(4)}"
    row = {"token": tok, "kind": "image", "model": model, "count": count,
           "card_id": card.get("card_id", ""), "issued_at": time.time(),
           "expires_at": time.time() + ttl, "consumed": False}
    (TOKENS / f"{tok}.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    audit("token_issue", token=tok, kind="image", model=model, count=count)
    return tok
