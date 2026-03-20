# memory/session_memory.py
# In-Memory Short-Term Memory for the Chronic Disease Management Agent v2.
#
# ══════════════════════════════════════════════════════════════════════════════
# WHY MEMORY FOR A CARE AGENT?
# ══════════════════════════════════════════════════════════════════════════════
#
# Without memory, every monitoring pass is isolated:
#   Pass 1: BP=165/95. High. Alert logged.
#   Pass 2: BP=168/97. No context. Same alert logged again.
#   Pass 3: BP=142/88. No idea if this is an improvement.
#
# With in-memory session context:
#   Pass 1: BP=165/95. Alert issued. Memory records: BP elevated, alert sent.
#   Pass 2: BP=168/97. Memory: "3rd consecutive high reading, trend WORSENING".
#           Planner escalates to urgent physician notification.
#   Pass 3: BP=142/88. Memory: "improving — responding to intervention".
#           Executor skips duplicate alert; logs improvement note.
#
# SCOPE: In-process memory per monitoring session.
# Suitable for MVP. For production: serialise to PostgreSQL/FHIR.

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class VitalSnapshot:
    timestamp:    str
    patient_id:   str
    max_glucose:  Optional[float]
    avg_bp_sys:   Optional[float]
    min_spo2:     Optional[float]
    adherence_avg:float
    symptoms:     List[str]
    risk_level:   str


@dataclass
class IssuedIntervention:
    intervention_id: str
    patient_id:      str
    type:            str
    urgency:         str
    title:           str
    recipient:       str
    issued_at:       str
    approved_by_human: bool = False


class SessionMemory:
    """
    Tracks vitals trends and intervention history within a monitoring session.
    One instance per patient, shared across all passes.
    """

    _URGENCY_RANK = {"routine": 1, "urgent": 2, "emergency": 3}

    def __init__(self, max_snapshots: int = 10):
        self.max_snapshots   = max_snapshots
        self.snapshots:       List[VitalSnapshot]        = []
        self.interventions:   List[IssuedIntervention]   = []
        self.pending_approvals: List[Dict[str, Any]]     = []
        self.session_start:   str = datetime.now().isoformat()
        self.pass_count:      int = 0

    def record_pass(self, state: dict) -> None:
        """Called at the end of each monitoring pass."""
        self.pass_count += 1
        pid      = state.get("patient_id", "unknown")
        glucose  = state.get("glucose_readings", [])
        bp       = state.get("bp_readings", [])
        vitals   = state.get("vitals_readings", [])
        meds     = state.get("medication_records", [])
        symptoms = state.get("symptom_reports", [])

        max_gluc = max((r["glucose_mgdl"] for r in glucose), default=None) if glucose else None
        avg_sys  = (round(sum(r["systolic_mmhg"] for r in bp) / len(bp), 1)
                    if bp else None)
        min_spo2 = min((r["spo2_pct"] for r in vitals), default=None) if vitals else None
        adh_avg  = (round(sum(m["adherence_pct"] for m in meds) / len(meds), 1)
                    if meds else 100.0)
        symp_names = [s["symptom"] for s in (symptoms or [])]

        snap = VitalSnapshot(
            timestamp    = datetime.now().isoformat(),
            patient_id   = pid,
            max_glucose  = max_gluc,
            avg_bp_sys   = avg_sys,
            min_spo2     = min_spo2,
            adherence_avg= adh_avg,
            symptoms     = symp_names,
            risk_level   = state.get("risk_level", "unknown"),
        )
        self.snapshots.append(snap)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]

    def record_intervention(self, intervention: dict) -> None:
        self.interventions.append(IssuedIntervention(
            intervention_id  = intervention.get("id", "INT-?"),
            patient_id       = intervention.get("patient_id", ""),
            type             = intervention.get("type", ""),
            urgency          = intervention.get("urgency", "routine"),
            title            = intervention.get("title", ""),
            recipient        = intervention.get("recipient", ""),
            issued_at        = datetime.now().isoformat(),
        ))

    def add_pending_approval(self, action: dict) -> None:
        """Queue an action that needs human approval before execution."""
        action["queued_at"] = datetime.now().isoformat()
        self.pending_approvals.append(action)

    def can_issue_intervention(self, intervention_type: str,
                               urgency: str, patient_id: str) -> bool:
        """
        Prevent duplicate interventions of the same type and urgency.
        Allow escalation to higher urgency.
        """
        matching = [
            i for i in self.interventions
            if i.type == intervention_type and i.patient_id == patient_id
        ]
        if not matching:
            return True
        last      = max(matching, key=lambda i: i.issued_at)
        last_rank = self._URGENCY_RANK.get(last.urgency, 0)
        new_rank  = self._URGENCY_RANK.get(urgency, 0)
        return new_rank > last_rank

    def get_trend(self) -> Dict[str, Any]:
        if len(self.snapshots) < 2:
            return {"available": False, "pass_count": self.pass_count}

        recent = self.snapshots[-1]
        prev   = self.snapshots[-2]

        bp_trend = "stable"
        if recent.avg_bp_sys and prev.avg_bp_sys:
            diff = recent.avg_bp_sys - prev.avg_bp_sys
            bp_trend = "worsening" if diff > 5 else "improving" if diff < -5 else "stable"

        gluc_trend = "stable"
        if recent.max_glucose and prev.max_glucose:
            diff = recent.max_glucose - prev.max_glucose
            gluc_trend = "worsening" if diff > 20 else "improving" if diff < -20 else "stable"

        consec_high_bp = 0
        for s in reversed(self.snapshots):
            if s.avg_bp_sys and s.avg_bp_sys > 140:
                consec_high_bp += 1
            else:
                break

        return {
            "available":         True,
            "pass_count":        self.pass_count,
            "bp_trend":          bp_trend,
            "glucose_trend":     gluc_trend,
            "consecutive_high_bp": consec_high_bp,
            "adherence_avg":     recent.adherence_avg,
            "active_symptoms":   recent.symptoms,
            "risk_level":        recent.risk_level,
        }

    def context_summary(self) -> str:
        trend = self.get_trend()
        n_int = len(self.interventions)

        if not trend["available"]:
            return (
                f"SESSION MEMORY: Pass #1 — no prior data. Establishing baseline."
            )

        lines = [
            f"SESSION MEMORY (pass #{self.pass_count}):",
            f"  BP trend      : {trend['bp_trend'].upper()}",
            f"  Glucose trend : {trend['glucose_trend'].upper()}",
            f"  Adherence avg : {trend['adherence_avg']}%",
            f"  Interventions : {n_int} issued this session",
        ]
        if trend["consecutive_high_bp"] >= 3:
            lines.append(
                f"  *** PERSISTENT HYPERTENSION: {trend['consecutive_high_bp']} "
                f"consecutive passes with avg SBP > 140 — escalate"
            )
        if trend["active_symptoms"]:
            lines.append(f"  Active symptoms: {', '.join(trend['active_symptoms'])}")
        if self.pending_approvals:
            lines.append(
                f"  Pending human approval: {len(self.pending_approvals)} action(s)"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self.__init__(max_snapshots=self.max_snapshots)


# Patient-keyed singletons
_MEMORIES: Dict[str, SessionMemory] = {}

def get_memory(patient_id: str) -> SessionMemory:
    if patient_id not in _MEMORIES:
        _MEMORIES[patient_id] = SessionMemory()
    return _MEMORIES[patient_id]
