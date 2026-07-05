"""Headroom -- hardware-aware agentic-RAG budget governor.

Consulted BEFORE each escalation in an agentic-RAG loop (an extra retrieval
hop, a bigger reranker, a longer-context pass). Instead of gating on a *token*
budget, Headroom gates on the MEASURED physical margin of the box right now:

  * VRAM headroom   -- how much GPU memory is free vs. what the next hop needs
  * thermal margin  -- degrees C of slack before a configurable abort temp
  * latency budget  -- observed per-hop latency vs. an allowed ceiling

It answers one question: "What can this box afford right *now*?" and returns
one of three decisions:

  ALLOW  -- enough margin on every axis; proceed with the escalation.
  DEFER  -- a soft limit is crossed (e.g. thermal is warm but not critical, or
            VRAM would dip below the safety margin). The caller should retry
            later / after a cooldown rather than abandon the task.
  DENY   -- a hard limit is crossed (VRAM cannot fit the next hop, or temp is at
            the abort line). Escalating now would risk an OOM or a thermal wedge.

Telemetry is read from an INJECTABLE source (a callable returning a Telemetry),
so the real nvidia-smi reader can be swapped for a deterministic fake in tests.
Every decision is logged with the telemetry that produced it.

STDLIB only. No network, no third-party deps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, List, Optional


class Decision(str, Enum):
    ALLOW = "allow"
    DEFER = "defer"
    DENY = "deny"


@dataclass
class Telemetry:
    """A single snapshot of GPU state (what a mock/real nvidia-smi would give)."""
    vram_used_mb: float          # currently used VRAM
    vram_total_mb: float         # total VRAM on the card
    temp_c: float                # current GPU core temperature
    last_hop_latency_ms: float   # latency of the most recent hop (0 if none yet)

    @property
    def vram_free_mb(self) -> float:
        return max(0.0, self.vram_total_mb - self.vram_used_mb)


@dataclass
class GovernorConfig:
    """Configurable margins. Defaults are tuned for a fragile 24GB card."""
    # Keep at least this many MB of VRAM free AFTER the next hop's estimated cost.
    vram_safety_margin_mb: float = 1024.0
    # Hard abort temperature; DENY at or above this.
    abort_temp_c: float = 83.0
    # Defer once we get within this many degrees of the abort temp.
    thermal_defer_margin_c: float = 5.0
    # Escalations that would push a hop past this latency are deferred.
    latency_budget_ms: float = 4000.0


@dataclass
class LogRecord:
    hop: int
    decision: Decision
    reason: str
    telemetry: Telemetry
    est_hop_cost_mb: float
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


class HeadroomGovernor:
    """Gates escalations on measured VRAM / thermal / latency margin."""

    def __init__(
        self,
        telemetry_source: Callable[[], Telemetry],
        config: Optional[GovernorConfig] = None,
    ) -> None:
        if not callable(telemetry_source):
            raise TypeError("telemetry_source must be a callable returning Telemetry")
        self._source = telemetry_source
        self.config = config or GovernorConfig()
        self.log: List[LogRecord] = []

    def _decide(self, tel: Telemetry, est_hop_cost_mb: float) -> tuple:
        cfg = self.config

        # --- HARD limits -> DENY -------------------------------------------
        # Thermal: at/above the abort line, never escalate.
        if tel.temp_c >= cfg.abort_temp_c:
            return (Decision.DENY,
                    "thermal: temp %.1fC >= abort %.1fC"
                    % (tel.temp_c, cfg.abort_temp_c))

        # VRAM: the next hop simply will not fit while keeping the safety margin.
        projected_free = tel.vram_free_mb - est_hop_cost_mb
        if projected_free < 0:
            return (Decision.DENY,
                    "vram: next hop needs %.0fMB but only %.0fMB free (OOM)"
                    % (est_hop_cost_mb, tel.vram_free_mb))

        # --- SOFT limits -> DEFER ------------------------------------------
        # Thermal: within the defer margin of the abort temp -> let it cool.
        thermal_defer_at = cfg.abort_temp_c - cfg.thermal_defer_margin_c
        if tel.temp_c >= thermal_defer_at:
            return (Decision.DEFER,
                    "thermal: temp %.1fC within %.1fC of abort %.1fC"
                    % (tel.temp_c, cfg.thermal_defer_margin_c, cfg.abort_temp_c))

        # VRAM: it fits, but would eat into the safety margin -> defer.
        if projected_free < cfg.vram_safety_margin_mb:
            return (Decision.DEFER,
                    "vram: post-hop free %.0fMB < safety margin %.0fMB"
                    % (projected_free, cfg.vram_safety_margin_mb))

        # Latency: the last hop already blew the budget -> defer more work.
        if tel.last_hop_latency_ms > cfg.latency_budget_ms:
            return (Decision.DEFER,
                    "latency: last hop %.0fms > budget %.0fms"
                    % (tel.last_hop_latency_ms, cfg.latency_budget_ms))

        # --- Everything in margin -> ALLOW ---------------------------------
        return (Decision.ALLOW,
                "ok: free %.0fMB, temp %.1fC, last %.0fms"
                % (tel.vram_free_mb, tel.temp_c, tel.last_hop_latency_ms))

    def check(self, hop: int, est_hop_cost_mb: float = 512.0) -> LogRecord:
        """Consult the governor before hop `hop`. Reads telemetry, decides, logs."""
        tel = self._source()
        decision, reason = self._decide(tel, est_hop_cost_mb)
        rec = LogRecord(
            hop=hop,
            decision=decision,
            reason=reason,
            telemetry=tel,
            est_hop_cost_mb=est_hop_cost_mb,
        )
        self.log.append(rec)
        return rec
