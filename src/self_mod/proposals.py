from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.self_mod.live_config import set_override
from src.self_mod.tier1_safe import is_tier1_eligible
from src.utils.logging import get_logger

log = get_logger("self_mod")

PROPOSALS_PATH = Path("data/proposals.json")
AUDIT_PATH = Path("data/self_mod_audit.json")


class Tier(int, Enum):
    AUTO = 1
    HITL = 2


class Proposal(BaseModel):
    id: str = ""
    target: str
    value: Any
    rationale: str
    tier: Tier
    status: str = "pending"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    decided_at: str | None = None
    decided_by: str | None = None


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _save_json(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _make_id(target: str, value: Any) -> str:
    raw = f"{target}|{value}|{datetime.utcnow().isoformat()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def apply_or_queue(target: str, value: Any, rationale: str) -> Proposal:
    """Главный entry-point. Если Tier-1 — применяет немедленно, иначе → pending Tier-2."""
    eligible, reason = is_tier1_eligible(target, value)
    pid = _make_id(target, value)

    if eligible:
        before = set_override(target, value)
        proposal = Proposal(
            id=pid, target=target, value=value, rationale=rationale,
            tier=Tier.AUTO, status="applied",
            decided_at=datetime.utcnow().isoformat(), decided_by="auto",
        )
        audit = _load_json(AUDIT_PATH)
        audit.append({
            "id": pid, "target": target, "before": before, "after": value,
            "rationale": rationale, "ts": datetime.utcnow().isoformat(),
        })
        _save_json(AUDIT_PATH, audit)
        log.info("self_mod.tier1_applied", target=target, before=before, after=value)
        return proposal

    proposal = Proposal(
        id=pid, target=target, value=value, rationale=f"{rationale} ({reason})",
        tier=Tier.HITL, status="pending",
    )
    proposals = _load_json(PROPOSALS_PATH)
    proposals.append(proposal.model_dump(mode="json"))
    _save_json(PROPOSALS_PATH, proposals)
    log.info("self_mod.tier2_queued", target=target, id=pid, reason=reason)
    return proposal


def list_audit(limit: int = 50) -> list[dict]:
    items = _load_json(AUDIT_PATH)
    return list(reversed(items))[:limit]


def list_proposals(status: str | None = None) -> list[Proposal]:
    raw = _load_json(PROPOSALS_PATH)
    if status:
        raw = [r for r in raw if r.get("status") == status]
    return [Proposal.model_validate(r) for r in raw]


def _decide(pid: str, decision: str, decided_by: str = "user") -> Proposal | None:
    raw = _load_json(PROPOSALS_PATH)
    for r in raw:
        if r.get("id") == pid and r.get("status") == "pending":
            r["status"] = decision
            r["decided_at"] = datetime.utcnow().isoformat()
            r["decided_by"] = decided_by
            _save_json(PROPOSALS_PATH, raw)
            log.info("self_mod.decision", id=pid, decision=decision, target=r.get("target"))
            return Proposal.model_validate(r)
    return None


def approve_proposal(pid: str, decided_by: str = "user") -> Proposal | None:
    """Одобрить Tier-2: меняет статус на 'approved'. Применение зависит от target —
    числовые параметры применяются через set_override, остальные требуют код-правки."""
    p = _decide(pid, "approved", decided_by)
    if p:
        # Любое одобренное предложение, чей target численно-применим (есть в TIER1_ALLOWLIST
        # с другим значением, или сторонний числовой target), пытаемся применить через
        # live_config. Текстовые/структурные правки требуют ручной правки кода.
        try:
            set_override(p.target, p.value)
            log.info("self_mod.tier2_applied_via_override", target=p.target, value=p.value)
        except (TypeError, ValueError) as e:
            log.warning("self_mod.tier2_apply_skipped", target=p.target, error=str(e))
    return p


def reject_proposal(pid: str, decided_by: str = "user") -> Proposal | None:
    return _decide(pid, "rejected", decided_by)
