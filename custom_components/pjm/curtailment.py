"""Elapsed-time curtailment accounting, independent of HA and prediction locks."""
from datetime import datetime, timedelta, timezone
import math


class CurtailmentBudget:
    """Count all active intervals against a local calendar-day allowance.

    Checkpoints include whether the output was active. On restart, conservatively
    charge downtime until now if it was active, then require a fresh decision.
    """

    def __init__(self):
        self.day = None
        self.seconds = 0.0
        self.active = False
        self.checkpoint = None
        self.last_activation = None

    def advance(self, now, requested, limit_seconds):
        if now.tzinfo is None:
            raise ValueError("Curtailment accounting needs timezone-aware time")
        today = now.date().isoformat()
        if today != self.day:
            self.day = today
            self.seconds = 0.0
            self.active = False
            self.checkpoint = None
            self.last_activation = None
        if self.active and self.checkpoint is not None:
            elapsed = max(0.0, (now.astimezone(timezone.utc) -
                                self.checkpoint.astimezone(timezone.utc)).total_seconds())
            self.seconds += elapsed
        self.checkpoint = now
        active = bool(requested and self.seconds < limit_seconds)
        if active and not self.active:
            self.last_activation = now.isoformat()
        self.active = active
        return active

    def deadline(self, now, limit_seconds):
        if not self.active:
            return None
        return now.astimezone(timezone.utc) + timedelta(seconds=max(0, limit_seconds-self.seconds))

    def snapshot(self):
        return dict(day=self.day, seconds=self.seconds, active=self.active,
                    checkpoint=self.checkpoint.isoformat() if self.checkpoint else None,
                    last_activation=self.last_activation)

    def restore(self, data, now, limit_seconds):
        if not data or data.get("day") != now.date().isoformat():
            self.advance(now, False, limit_seconds)
            return
        seconds = float(data.get("seconds", 0))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid saved curtailment usage")
        self.day = data["day"]
        self.seconds = seconds
        raw = data.get("checkpoint")
        self.checkpoint = datetime.fromisoformat(raw) if raw else now
        if self.checkpoint.tzinfo is None:
            raise ValueError("Invalid saved curtailment checkpoint")
        self.active = bool(data.get("active"))
        self.last_activation = data.get("last_activation")
        self.advance(now, False, limit_seconds)
