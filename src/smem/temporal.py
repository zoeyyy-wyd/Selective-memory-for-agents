"""Temporal constraint parsing and interval algebra (plan section 07).

A question's temporal constraint becomes a closed interval [t1, t2]; a fact's validity is
[valid_from, valid_to); the chain node whose interval intersects the constraint is selected.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

_NOW_WORDS = re.compile(r"\b(now|currently|current|these days|at the moment|today|nowadays|present)\b", re.IGNORECASE)
_BEFORE_LATEST = re.compile(r"\b(last time|previously|before that|the previous|used to|earlier|originally|at first|initially)\b", re.IGNORECASE)
_CHANGE = re.compile(r"\b(how many times|how often|each time|every time|all the|history of|changed|changes|over time|so far|in total|different)\b", re.IGNORECASE)
_AGO = re.compile(r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?\s+ago", re.IGNORECASE)
_LAST_UNIT = re.compile(r"\b(last|past|previous)\s+(week|month|year)\b", re.IGNORECASE)
_MONTH = re.compile(r"\b(in|during|around|of|on|since|from|by)?\s*(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b\.?\s*(\d{4})?", re.IGNORECASE)
_DATE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_WEEKDAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)

_WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}


@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime  # inclusive

    def intersects(self, start: datetime, end: datetime | None) -> bool:
        """Does this interval intersect the half-open validity [start, end)?"""
        if end is None:
            return self.end >= start
        return self.end >= start and self.start < end

    def contains(self, t: datetime) -> bool:
        return self.start <= t <= self.end


@dataclass
class TemporalConstraint:
    interval: Interval | None = None
    mode: str = "none"       # none | now | interval | before_latest | whole_chain
    raw: str = ""

    @property
    def is_now(self) -> bool:
        return self.mode == "now"


def _month_interval(year: int, month: int) -> Interval:
    last = calendar.monthrange(year, month)[1]
    return Interval(datetime(year, month, 1), datetime(year, month, last, 23, 59, 59))


def _shift_months(t: datetime, months: int) -> datetime:
    total = t.year * 12 + (t.month - 1) - months
    y, m = divmod(total, 12)
    m += 1
    d = min(t.day, calendar.monthrange(y, m)[1])
    return t.replace(year=y, month=m, day=d)


def parse_temporal(question: str, now: datetime) -> TemporalConstraint:
    q = question.strip()

    if _CHANGE.search(q):
        return TemporalConstraint(None, "whole_chain", q)

    m = _DATE.search(q)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        day = datetime(y, mo, d)
        return TemporalConstraint(Interval(day, day + timedelta(days=1) - timedelta(seconds=1)), "interval", q)

    m = _AGO.search(q)
    if m:
        n_raw, unit = m.group(1).lower(), m.group(2).lower()
        n = int(n_raw) if n_raw.isdigit() else _WORD_NUM[n_raw]
        if unit == "day":
            centre = now - timedelta(days=n)
            return TemporalConstraint(Interval(centre - timedelta(days=1), centre + timedelta(days=1)), "interval", q)
        if unit == "week":
            centre = now - timedelta(weeks=n)
            return TemporalConstraint(Interval(centre - timedelta(days=3), centre + timedelta(days=3)), "interval", q)
        if unit == "month":
            centre = _shift_months(now, n)
            return TemporalConstraint(_month_interval(centre.year, centre.month), "interval", q)
        centre = now.replace(year=now.year - n)
        return TemporalConstraint(Interval(datetime(centre.year, 1, 1), datetime(centre.year, 12, 31, 23, 59, 59)), "interval", q)

    m = _LAST_UNIT.search(q)
    if m:
        unit = m.group(2).lower()
        if unit == "week":
            return TemporalConstraint(Interval(now - timedelta(days=14), now), "interval", q)
        if unit == "month":
            prev = _shift_months(now, 1)
            return TemporalConstraint(_month_interval(prev.year, prev.month), "interval", q)
        return TemporalConstraint(Interval(datetime(now.year - 1, 1, 1), datetime(now.year - 1, 12, 31, 23, 59, 59)), "interval", q)

    m = _MONTH.search(q)
    if m and m.group(2).lower() in MONTHS and (len(m.group(2)) > 3 or m.group(2).istitle()):
        month = MONTHS[m.group(2).lower()]
        if m.group(3):
            year = int(m.group(3))
        else:
            # most recent occurrence of that month at or before `now`
            year = now.year if month <= now.month else now.year - 1
        return TemporalConstraint(_month_interval(year, month), "interval", q)

    if _NOW_WORDS.search(q):
        return TemporalConstraint(Interval(now, now), "now", q)
    if _BEFORE_LATEST.search(q):
        return TemporalConstraint(None, "before_latest", q)
    return TemporalConstraint(None, "none", q)


def parse_session_date(text: str) -> datetime:
    """LongMemEval timestamps look like '2023/05/20 (Sat) 02:21'. Falls back to dateparser."""
    text = text.strip()
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    import dateparser

    parsed = dateparser.parse(text)
    if parsed is not None:
        return parsed.replace(tzinfo=None)
    raise ValueError(f"unrecognised date: {text!r}")


def normalize_relative_date(text: str, session_ts: datetime) -> datetime | None:
    """Resolve 'last Wednesday' / 'two weeks ago' / 'yesterday' relative to the session timestamp.
    Used at extraction time so stored facts carry absolute dates."""
    try:
        import dateparser
    except ImportError:  # pragma: no cover
        return None
    prefer = "past"
    cleaned = text.strip()
    # dateparser resolves a bare weekday relative to the base but not "last Wednesday" / "next Friday"
    m = re.match(r"^(last|this|past|next|coming)\s+(" + "|".join(calendar.day_name).lower() + r")\b", cleaned, re.IGNORECASE)
    if m:
        prefer = "future" if m.group(1).lower() in ("next", "coming") else "past"
        cleaned = cleaned[m.end(1):].strip()
    parsed = dateparser.parse(
        cleaned,
        settings={"RELATIVE_BASE": session_ts, "PREFER_DATES_FROM": prefer, "RETURN_AS_TIMEZONE_AWARE": False},
    )
    return parsed
