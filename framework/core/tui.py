"""autoresearch TUI 辅助函数 — 训练过程可视化。

纯渲染工具，从 train.py 中提取。不依赖训练状态 — 仅将数据转换为字符串。
"""


def _fit_series(values: list[float], width: int) -> list[float]:
    """Return exactly width samples for stable chart boundaries."""
    if not values or width <= 0:
        return []

    recent = list(values[-width:])
    if len(recent) == width:
        return recent
    if len(recent) == 1:
        return recent * width

    last = len(recent) - 1
    scaled: list[float] = []
    for index in range(width):
        pos = index * last / (width - 1)
        lo = int(pos)
        hi = min(lo + 1, last)
        frac = pos - lo
        scaled.append(recent[lo] * (1.0 - frac) + recent[hi] * frac)
    return scaled


def sparkline(values: list[float], width: int = 30) -> str:
    """单行火花图，用于紧凑展示趋势。"""
    if not values:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    recent = _fit_series(values, width)
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi > lo else 1.0
    return "".join(chars[min(int((v - lo) / span * 7), 7)] for v in recent)


def sparkline2(values: list[float], width: int = 40) -> tuple[str, str]:
    """双行火花图 — 返回 (upper_row, lower_row)。

    将值映射到 0-15 级，分两行字符显示：
    - 下行: ▁▂▃▄▅▆▇█ 表示 0-7 级
    - 上行: 表示 8-15 级（下行显示实心 █）
    """
    if not values:
        return ("", "")
    lo_chars = " ▁▂▃▄▅▆▇"  # 0-7: 下半
    hi_chars = " ▁▂▃▄▅▆▇"  # 8-15: 上半
    recent = _fit_series(values, width)
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


def sparkline3(values: list[float], width: int = 40) -> tuple[str, str, str]:
    """三行火花图 — 返回 (upper_row, middle_row, lower_row)。

    将值映射到 0-23 级，分三行字符显示：
    - 下行: ▁▂▃▄▅▆▇█ 表示 0-7 级
    - 中行: 表示 8-15 级（下行为实心 █）
    - 上行: 表示 16-23 级（下行+中行为实心 █）
    """
    if not values:
        return ("", "", "")

    chars = " ▁▂▃▄▅▆▇"
    recent = _fit_series(values, width)
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi > lo else 1.0

    upper = []
    middle = []
    lower = []
    for v in recent:
        level = min(int((v - lo) / span * 23), 23)
        if level >= 16:
            lower.append("█")
            middle.append("█")
            upper.append(chars[level - 16])
        elif level >= 8:
            lower.append("█")
            middle.append(chars[level - 8])
            upper.append(" ")
        else:
            lower.append(chars[level])
            middle.append(" ")
            upper.append(" ")

    return ("".join(upper), "".join(middle), "".join(lower))


def sparkline4(values: list[float], width: int = 40) -> tuple[str, str, str, str]:
    """四行火花图 — 返回 (top, upper_mid, lower_mid, bottom)。

    将值映射到 0-31 级，分四行字符显示，每行使用 8 级半块字符。
    """
    if not values:
        return ("", "", "", "")

    chars = " ▁▂▃▄▅▆▇"
    recent = _fit_series(values, width)
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi > lo else 1.0

    top = []
    upper_mid = []
    lower_mid = []
    bottom = []
    for v in recent:
        level = min(int((v - lo) / span * 31), 31)
        if level >= 24:
            bottom.append("█")
            lower_mid.append("█")
            upper_mid.append("█")
            top.append(chars[level - 24])
        elif level >= 16:
            bottom.append("█")
            lower_mid.append("█")
            upper_mid.append(chars[level - 16])
            top.append(" ")
        elif level >= 8:
            bottom.append("█")
            lower_mid.append(chars[level - 8])
            upper_mid.append(" ")
            top.append(" ")
        else:
            bottom.append(chars[level])
            lower_mid.append(" ")
            upper_mid.append(" ")
            top.append(" ")

    return (
        "".join(top),
        "".join(upper_mid),
        "".join(lower_mid),
        "".join(bottom),
    )


def progress_bar(elapsed: float, budget: float | None, width: int = 34) -> str:
    """渲染文本进度条，显示已用时间/总时间。"""
    if budget is None or budget <= 0:
        return ""
    frac = min(elapsed / budget, 1.0)
    filled = int(frac * width)
    bar = "━" * filled + "─" * (width - filled)
    em, es = divmod(int(elapsed), 60)
    bm, bs = divmod(int(budget), 60)
    return f"{bar} {frac:3.0%} {em}:{es:02d} / {bm}:{bs:02d}"
