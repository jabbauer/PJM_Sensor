"""Timestamp-preserving PJM day-ahead price parsing."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def interval_start(item):
    """Prefer unambiguous UTC; reject ambiguous/missing Eastern-only times."""
    raw = item.get("datetime_beginning_utc")
    if raw:
        try:
            stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None
    raw = item.get("datetime_beginning_ept")
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(str(raw))
        if stamp.tzinfo is not None:
            return stamp.astimezone(timezone.utc)
        first, second = stamp.replace(tzinfo=EASTERN, fold=0), stamp.replace(tzinfo=EASTERN, fold=1)
        if first.utcoffset() != second.utcoffset():
            return None
        return first.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def parse_prices(items, zone=None, prefer_pnode_id=None, local_timezone=timezone.utc):
    """Keep distinct UTC intervals; local 0..23 attributes are compatibility views.

    A repeated local hour has no single numeric compatibility value. Consumers
    should use the offset-bearing hourly_prices list, which preserves both.
    """
    buckets = defaultdict(list)
    zone_u = (zone or "").upper()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        item_zone = str(item.get("zone") or "").upper()
        if zone_u and item_zone and zone_u != item_zone:
            continue
        if str(item.get("row_is_current", "TRUE")).upper() in ("FALSE", "0", "N", "NO"):
            continue
        try:
            value = float(item["total_lmp_da"])
            if not math.isfinite(value):
                continue
            pnode = int(item["pnode_id"]) if item.get("pnode_id") is not None else None
        except (KeyError, TypeError, ValueError):
            continue
        stamp = interval_start(item)
        if stamp is None:
            continue
        if prefer_pnode_id is not None and pnode != int(prefer_pnode_id):
            continue
        name = str(item.get("pnode_name") or "")
        priority = 0 if prefer_pnode_id is not None else (
            1 if "ZONE" in str(item.get("type") or "").upper() else
            2 if zone_u and zone_u in name.upper() else 3)
        buckets[stamp].append((priority, value, name))
    intervals = []
    by_hour = {}
    names = Counter()
    dates = Counter()
    for stamp, values in sorted(buckets.items()):
        best = min(v[0] for v in values)
        chosen = [v for v in values if v[0] == best]
        value = sum(v[1] for v in chosen) / len(chosen)
        local = stamp.astimezone(local_timezone)
        intervals.append(dict(start_utc=stamp.isoformat(), start_local=local.isoformat(), price=value))
        by_hour[local.hour] = value if local.hour not in by_hour else None
        dates[local.date()] += 1
        names.update(v[2] for v in chosen if v[2])
    return dict(by_hour=by_hour, intervals=intervals,
                lmp_day=dates.most_common(1)[0][0] if dates else None,
                pnode_name=names.most_common(1)[0][0] if names else None)
