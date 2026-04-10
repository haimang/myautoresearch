"""TUI helper functions for MAG-Gomoku training display.

Pure rendering utilities extracted from train.py. These have no
dependencies on training state — they just transform data into strings.
"""


def sparkline(values: list[float], width: int = 30) -> str:
    """Single-row sparkline for compact display."""
    if not values:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi > lo else 1.0
    return "".join(chars[min(int((v - lo) / span * 7), 7)] for v in recent)


def sparkline2(values: list[float], width: int = 40) -> tuple[str, str]:
    """Double-height sparkline — returns (upper_row, lower_row).

    Maps values to 0-15 levels across two character rows:
    - Lower row: blocks ▁▂▃▄▅▆▇█ for levels 0-7
    - Upper row: blocks for levels 8-15 (lower row shows solid █)
    """
    if not values:
        return ("", "")
    lo_chars = " ▁▂▃▄▅▆▇"  # 0-7: lower half
    hi_chars = " ▁▂▃▄▅▆▇"  # 8-15: upper half (shifted)
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi > lo else 1.0
    upper = []
    lower = []
    for v in recent:
        level = min(int((v - lo) / span * 15), 15)
        if level >= 8:
            lower.append("█")
            upper.append(hi_chars[level - 8])
        else:
            lower.append(lo_chars[level])
            upper.append(" ")
    return ("".join(upper), "".join(lower))


def progress_bar(elapsed: float, budget: float | None, width: int = 34) -> str:
    """Render a text progress bar with time fraction."""
    if budget is None or budget <= 0:
        return ""
    frac = min(elapsed / budget, 1.0)
    filled = int(frac * width)
    bar = "━" * filled + "─" * (width - filled)
    em, es = divmod(int(elapsed), 60)
    bm, bs = divmod(int(budget), 60)
    return f"{bar} {frac:3.0%} {em}:{es:02d} / {bm}:{bs:02d}"
