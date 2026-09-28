"""Small shared utilities: hashing, chunking, humanising, canonical JSON."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

BLAKE_CHUNK = 1 << 20  # 1 MiB


def file_hash(path: Path, size_hint: int | None = None) -> str:
    """Streaming blake2b of a file (16-byte digest, hex)."""
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while True:
            b = f.read(BLAKE_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(obj) -> str:
    """Deterministic hash of any JSON-serialisable object (cache keys)."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=12).hexdigest()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_jsonc(text: str) -> str:
    """Remove `//` and `/* */` comments outside JSON strings.

    The README documents cards in JSONC (its architecture example carries a
    `// usd = …` comment), so a card copied from the documentation must load.
    Comment characters are replaced by spaces, keeping line/column positions
    of any later parse error meaningful."""
    out, i, n = [], 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def load_jsonc(path: Path):
    """json.load that accepts JSONC comments (cards, run cards, batch meta)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(strip_jsonc(text))


def as_lang_list(value) -> list[str]:
    """A language declaration as a list: accepts "en", ["en"], None."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v]


def declared_langs(obj: dict | None) -> list[str]:
    """Languages declared on a card entry, section or annotation.

    The contract spells it `languages` in cards and `language` in documents;
    users mix both, so both are accepted everywhere."""
    if not obj:
        return []
    return as_lang_list(obj.get("languages") or obj.get("language"))


def line_chunks(path: Path, target: int) -> list[tuple[int, int]]:
    """Split a file into byte ranges of roughly `target` bytes, aligned on
    newline boundaries, so each range holds only whole lines."""
    size = path.stat().st_size
    if size == 0:
        return []
    if size <= target:
        return [(0, size)]
    ranges: list[tuple[int, int]] = []
    with open(path, "rb") as f:
        start = 0
        while start < size:
            end = min(start + target, size)
            if end < size:
                f.seek(end)
                f.readline()  # advance to the end of the current line
                end = f.tell()
            ranges.append((start, min(end, size)))
            start = end
    return ranges


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def human_duration(s: float | None) -> str:
    if s is None or not math.isfinite(s):
        return "—"
    s = max(0.0, float(s))
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m} min {sec:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


# a cost unit as a column header (the card's unit tokens are for machines)
UNIT_HEAD = {"usd": "Cost (USD)", "kwh": "Energy (kWh)", "time": "Time"}


def human_cost(unit: str, value) -> str:
    """A cost as a human reads it: $0.09 · 0.39 kWh · 1 hr 07 min.

    Raw floats (0.0898 usd, 3989.9 s) are how the model computes; they are not
    how a reader compares runs."""
    if value is None:
        return "—"
    u = (unit or "").lower()
    if u == "usd":
        v = float(value)
        if v and abs(v) < 0.01:
            return f"${v:.4f}".rstrip("0").rstrip(".")
        return f"${v:,.2f}".replace(",", " ")
    if u == "kwh":
        v = float(value)
        return f"{v:.2f} kWh" if abs(v) >= 0.01 else f"{v:.4g} kWh"
    if u in ("time", "s", "seconds"):
        return human_clock(value)
    return f"{fmt_num(value, 3)} {unit}"


def human_clock(s) -> str:
    """Wall time in reading units: 42 s · 7 min 12 s · 1 hr 07 min · 2 d 03 hr."""
    if s is None or not math.isfinite(float(s)):
        return "—"
    s = max(0.0, float(s))
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m} min {sec:02d} s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h} hr {m:02d} min"
    d, h = divmod(h, 24)
    return f"{d} d {h:02d} hr"


def mean_sd(values, digits: int = 1, pct: bool = False) -> str:
    """"4.9 ± 2.1" / "73.7% ± 18.4%" — the spread belongs next to the mean."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return "—"
    n = len(vals)
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    if pct:
        return f"{100 * mu:.{digits}f}% ± {100 * sd:.{digits}f}%"
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"


def human_count(n) -> str:
    if n is None:
        return "—"
    n = float(n)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} B"
    if abs(n) >= 100_000_000:
        return f"{n / 1_000_000:.0f} M"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if abs(n) >= 10_000:
        return f"{n / 1000:.1f} k"
    return f"{n:,.0f}"


def fmt_num(x, digits: int = 3) -> str:
    """Compact numeric formatting for labels/tables."""
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf == 0:
        return "0"
    if abs(xf) < 10 ** -digits or abs(xf) >= 10 ** 7:
        return f"{xf:.2e}"
    if xf == int(xf) and abs(xf) < 10 ** 7:
        return f"{int(xf):,}"
    return f"{xf:.{digits}f}".rstrip("0").rstrip(".")


class RateEMA:
    """Exponential moving average of a throughput (units per second)."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.rate: float | None = None
        self._t = time.monotonic()
        self._acc = 0.0

    def add(self, units: float) -> None:
        now = time.monotonic()
        self._acc += units
        dt = now - self._t
        if dt >= 0.5:  # resample every half second
            inst = self._acc / dt
            self.rate = inst if self.rate is None else (
                self.alpha * inst + (1 - self.alpha) * self.rate)
            self._t, self._acc = now, 0.0

    def eta(self, remaining_units: float) -> float | None:
        if not self.rate or self.rate <= 0:
            return None
        return remaining_units / self.rate
