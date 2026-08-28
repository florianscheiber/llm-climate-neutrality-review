# scripts/08-stage3c-electricity-price.py  (stage 3c -- exploratory analysis)
#
# What wholesale / system electricity price does a scenario carry in the year
# it reaches net zero?
#
# Net-zero scenarios come from netzero-flags.csv (is_netzero == True) -- the
# SAME filter as the carbon-price and sector-balance figures.
#
# For every net-zero scenario, the electricity-price rows of
# harmonized-costs.csv (Cost_Type == "Electricity price", value already in
# USD2025/MWh) at the pathway's last reported year are collected, reduced to
# ONE value per (paper, scenario) -- the median across every area / row at
# that year -- and compared in a box + ranked strip plot. Points are
# coloured by paper, because the spread is mostly a study-to-study effect.
#
# Technology-specific on-grid tariffs, feed-in tariffs, PPA / LCOE rows and a
# mislabelled food-price series are dropped (DROP_DETAIL) -- only a
# system / wholesale / retail electricity price is kept.
#
# Every number in the figure and the generated doc is computed here; nothing
# is hard-coded in the HTML.
#
# INPUT   outputs/output-stage3c/harmonized-costs.csv
#         outputs/output-stage3c/netzero-flags.csv
# OUTPUT  outputs/analysis/electricity-price-at-netzero.csv
#         figures/electricity-price-at-netzero.png
#         docs/electricity-price-at-netzero.html
#
#   python3 scripts/08-stage3c-electricity-price.py     (run from the repo root)

from pathlib import Path
import argparse
import base64
import csv
import html
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_common import (canon_scenario, canon_area,   # noqa: E402
                              is_catchall_scenario, resolve_scenario)

INK, INK2, MUTED, GRID = "#1a1a1a", "#55534e", "#8a8a84", "#dedcd5"

# Data points are coloured by world region. Fixed order and a colourblind-safe
# hue per region (Okabe-Ito based); only the regions present get a legend
# entry. Shared verbatim with 09-stage3c-capex-by-technology.py.
REGION_ORDER = [
    "North America", "Latin America", "Europe", "Africa", "Middle East",
    "China", "India", "Rest of Asia-Pacific", "Global / multi-region",
    "Not reported",
]
REGION_COLOR = {
    "North America":         "#0072b2",
    "Latin America":         "#b07d2b",
    "Europe":                "#56b4e9",
    "Africa":                "#cc79a7",
    "Middle East":           "#009e73",
    "China":                 "#d55e00",
    "India":                 "#e69f00",
    "Rest of Asia-Pacific":  "#7a5195",
    "Global / multi-region": "#8a8a84",
    "Not reported":          "#c2c0b8",
}

# (region, keyword pattern) -- first match wins, so specifics precede
# continents and the "global" aggregate.
_REGION_RULES = [
    ("Global / multi-region",
     r"\b(global|world(wide)?|multi-?region|gcam regions|all regions|"
     r"model regions|whole model|aggregate)\b"),
    ("China",
     r"\bchina\b|\bprc\b|chinese|nanning|shanxi|guangdong|guangxi|"
     r"beijing|shanghai|shenzhen|sichuan|inner mongolia|yangtze|"
     r"^(ea|na|sa|wa|nc|sc|ec|wc)$"),   # p137 China grid-region codes
    ("India", r"\bindia\b|indian\b|\bdelhi\b"),
    ("Middle East",
     r"\bsaudi|\buae\b|emirates|\bqatar|\bkuwait|\boman\b|\bbahrain|"
     r"\biran\b|\biraq\b|israel|\bjordan|middle east|\bgcc\b|persian gulf|"
     r"mena\b"),
    ("Africa",
     r"\bafrica|nigeria|\begypt|kenya|\bghana|ethiopia|morocco|tanzania|"
     r"\bsub-?saharan"),
    ("Europe",
     r"\b(europe|european union|eu\+?|euro area|portugal|spain|france|"
     r"germany|german|italy|poland|polish|voivodeship|switzerland|swiss|"
     r"austria|belgium|netherlands|dutch|norway|sweden|finland|denmark|"
     r"ireland|united kingdom|\buk\b|britain|scotland|wales|greece|"
     r"czech|slovak|hungary|romania|bulgaria|croatia|serbia|bavaria|"
     r"catalonia)\b"),
    ("North America",
     r"\b(united states|\bu\.?s\.?a?\.?\b|america\b|canada|canadian|"
     r"quebec|mexico|california|texas|new york|bay area|western us|"
     r"pjm|ercot|wecc|caiso)\b"),
    ("Latin America",
     r"\b(brazil|brazilian|chile|chilean|argentina|colombia|peru|"
     r"uruguay|latin america|south america)\b"),
    ("Rest of Asia-Pacific",
     r"\b(japan|japanese|korea|korean|australia|australian|new zealand|"
     r"thailand|thai|vietnam|indonesia|malaysia|philippines|singapore|"
     r"taiwan|asean|asia|pacific|oceania)\b"),
]


def world_region(area):
    a = (area or "").strip()
    if not a or a.lower() in ("not reported", "n/a", "unspecified",
                              "unknown", "none"):
        return "Not reported"
    for name, pat in _REGION_RULES:
        if re.search(pat, a, re.IGNORECASE):
            return name
    return "Not reported"

PRICE_TYPE = "Electricity price"

# Cost_Type_Detail phrasings that are NOT a system / wholesale / retail
# electricity price -- a single-technology on-grid tariff, a feed-in tariff,
# a PPA or LCOE quote, or the mislabelled staple-food series in one paper.
DROP_DETAIL = re.compile(
    r"food price"
    r"|on-?grid (wind|pv|photovolt|solar|hydro)"
    r"|feed-?in"
    r"|\bppa\b"
    r"|\blcoe\b"
    r"|levelis|levelized",
    re.IGNORECASE,
)

_NO_CDR = re.compile(r"\bno[\s_-]*(beccs|cdr|ccs|dac|net|removal)", re.IGNORECASE)

# a flag whose neutrality year is earlier than this is not a long-run
# net-zero end state
MIN_ENDPOINT_YEAR = 2030

DEFAULT_COSTS = Path("outputs/output-stage3c/harmonized-costs.csv")
DEFAULT_FLAGS = Path("outputs/output-stage3c/netzero-flags.csv")
DEFAULT_OUTDIR = Path("outputs/analysis")
DEFAULT_FIGDIR = Path("figures")
DEFAULT_DOC = Path("docs/electricity-price-at-netzero.html")


# ============================================================================
# LOADING / BUILDING
# ============================================================================

def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_year(raw):
    s = (raw or "").strip()
    if re.fullmatch(r"(?:19|20)\d{2}", s):
        return int(s)
    years = re.findall(r"(?:19|20)\d{2}", s)
    return int(years[-1]) if years else None


def source_ref(row):
    st = (row.get("Source_Type") or "").strip()
    sn = (row.get("Source_Number") or "").strip()
    if st and sn:
        return f"{st} {sn}"
    return st or sn or "Not reported"


_SUBNATIONAL = re.compile(
    r"\b(province|voivodeship|prefecture|county|municipal|canton|region|grid|"
    r"bay area|greater|quebec|nanning|shanxi|guangdong|guangxi|beijing|shenzhen|"
    r"shanghai|california|texas|wales|scotland|bavaria|catalonia|"
    r"[nesw]a\b|north|south|east|west)\b", re.IGNORECASE)


def geographic_scope(area):
    a = (area or "").strip()
    if not a:
        return "unspecified"
    if re.search(r"\b(global|world|worldwide|eu\+?|europe|gcam regions)\b",
                 a, re.IGNORECASE):
        return "supranational"
    if _SUBNATIONAL.search(a):
        return "subnational"
    return "national"


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _flag_index(flags):
    by_paper = {}
    for r in flags:
        y = parse_year(r.get("last_year"))
        if y is None or y < MIN_ENDPOINT_YEAR:
            continue
        sc = r.get("scenario_canon") or canon_scenario(r.get("Scenario_Name"))
        d = by_paper.setdefault(r["paper_id"], {})
        d[sc] = max(d.get(sc, 0), y)
    return by_paper


def _at_year(items, endpoint):
    """items: [(year, value, area, row)]. Rows at the year closest to
    `endpoint` (ties -> the later year); all rows when no year is known."""
    yrs = sorted({y for y, _, _, _ in items if y is not None})
    if not yrs:
        return items, None
    pick = min(yrs, key=lambda y: (abs(y - endpoint), -y))
    return [it for it in items if it[0] == pick], pick


def build(cost_rows, flags):
    flag_by_paper = _flag_index(flags)

    by_scenario, catchall, dropped_detail = {}, {}, 0
    for r in cost_rows:
        if (r.get("Cost_Type") or "").strip() != PRICE_TYPE:
            continue
        if (r.get("Value_std_Unit") or "").strip() != "USD2025/MWh":
            continue
        if DROP_DETAIL.search(r.get("Cost_Type_Detail", "")):
            dropped_detail += 1
            continue
        value = to_float(r.get("Value_std"))
        if value is None:
            continue
        pid = r["paper_id"]
        item = (parse_year(r.get("Year")), value, (r.get("Area") or "").strip(), r)
        if is_catchall_scenario(r.get("Scenario_Name")):
            catchall.setdefault(pid, []).append(item)
        else:
            sc = r.get("Scenario_canon") or canon_scenario(r.get("Scenario_Name"))
            by_scenario.setdefault((pid, sc), []).append(item)

    def _emit(pid, scen_disp, endpoint, items, via):
        used, pick = _at_year(items, endpoint)
        at_year = [v for _, v, _, _ in used]
        first = min(v for _, v, _, _ in items) if len(items) > 1 else None
        areas = sorted({a for _, _, a, _ in used if a})
        regions = {world_region(a) for a in areas} or {"Not reported"}
        region = regions.pop() if len(regions) == 1 else "Global / multi-region"
        meta = used[0][3]
        return {
            "paper_id": pid,
            "title": meta.get("Title", ""),
            "doi": meta.get("DOI", ""),
            "scenario": scen_disp,
            "area": areas[0] if len(areas) == 1 else f"{len(areas)} regions"
            if areas else "not reported",
            "areas_detail": " / ".join(areas),
            "region": region,
            "scope": geographic_scope(areas[0] if areas else ""),
            "netzero_year": endpoint,
            "price_year": pick if pick is not None else "",
            "matched_via": via,
            "n_price_rows": len(at_year),
            "price": round(statistics.median(at_year), 1),
            "price_range": (f"{round(min(at_year),1)}–{round(max(at_year),1)}"
                            if max(at_year) - min(at_year) > 0.5 else ""),
            "price_first_seen": round(first, 1) if first is not None else "",
            "raw_values": " | ".join(sorted({
                f"{(x.get('Value') or '').strip()} "
                f"{(x.get('Value_Unit') or '').strip()}".strip()
                for _, _, _, x in used})),
            "source": " | ".join(sorted({source_ref(x) for _, _, _, x in used})),
        }

    rows, matched = [], set()
    for (pid, sc), items in sorted(by_scenario.items()):
        per_paper = flag_by_paper.get(pid)
        if not per_paper:
            continue
        end = per_paper.get(sc)
        if end is None:
            end = resolve_scenario(sc, per_paper)
        if end is None:
            continue
        rows.append(_emit(pid, (items[0][3].get("Scenario_Name") or "").strip(),
                          end, items, "scenario"))
        matched.add((pid, sc))

    applied_catchall = 0
    disp = {}
    for r in flags:
        sc = r.get("scenario_canon") or canon_scenario(r.get("Scenario_Name"))
        disp.setdefault((r["paper_id"], sc), (r.get("Scenario_Name") or sc))
    for pid, items in sorted(catchall.items()):
        per_paper = flag_by_paper.get(pid)
        if not per_paper:
            continue
        for sc, end in sorted(per_paper.items()):
            if (pid, sc) in matched:
                continue
            rows.append(_emit(pid, disp.get((pid, sc), sc), end, items,
                              "paper-wide"))
            applied_catchall += 1

    rows.sort(key=lambda r: r["price"])
    return rows, {"dropped_detail": dropped_detail, "flags_total": len(flags),
                  "applied_catchall": applied_catchall}


# ============================================================================
# STATISTICS
# ============================================================================

def compute_stats(rows, meta):
    prices = [r["price"] for r in rows]
    q1, _, q3 = statistics.quantiles(prices, n=4)
    papers = sorted({r["paper_id"] for r in rows})

    by_paper = {}
    for r in rows:
        by_paper.setdefault(r["paper_id"], []).append(r)
    dom_id, dom_rows = max(by_paper.items(), key=lambda kv: len(kv[1]))
    dom_prices = sorted(r["price"] for r in dom_rows)

    low = rows[0]
    high = rows[-1]

    # BECCS-off vs BECCS-on pair, if any paper has one
    beccs = None
    for pid, prs in by_paper.items():
        off = [r for r in prs if _NO_CDR.search(r["scenario"])]
        on = [r for r in prs if not _NO_CDR.search(r["scenario"])]
        if off and on:
            beccs = {"paper": pid,
                     "with": statistics.median([r["price"] for r in on]),
                     "without": statistics.median([r["price"] for r in off])}
            beccs["delta"] = beccs["without"] - beccs["with"]
            break

    by_region = {}
    for r in rows:
        by_region.setdefault(r["region"], []).append(r["price"])
    regions = [g for g in REGION_ORDER if g in by_region]
    region_median = {g: round(statistics.median(by_region[g])) for g in regions}

    return {
        "n": len(rows),
        "n_papers": len(papers),
        "papers": papers,
        "regions": regions,
        "region_median": region_median,
        "region_n": {g: len(by_region[g]) for g in regions},
        "flags_total": meta["flags_total"],
        "dropped_detail": meta["dropped_detail"],
        "applied_catchall": meta["applied_catchall"],
        "p_min": round(min(prices)), "p_max": round(max(prices)),
        "median": round(statistics.median(prices)),
        "q1": round(q1), "q3": round(q3),
        "spread": round(max(prices) / max(min(prices), 1e-9), 1),
        "year_min": min(r["netzero_year"] for r in rows),
        "year_max": max(r["netzero_year"] for r in rows),
        "dom_paper": dom_id, "dom_n": len(dom_rows),
        "dom_area": dom_rows[0]["area"],
        "dom_min": round(dom_prices[0]), "dom_max": round(dom_prices[-1]),
        "dom_pct": round(100 * len(dom_rows) / len(rows)),
        "low_paper": low["paper_id"], "low_area": low["area"],
        "low_price": round(low["price"]),
        "high_paper": high["paper_id"], "high_area": high["area"],
        "high_price": round(high["price"]),
        "beccs": beccs,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["paper_id", "title", "doi", "scenario", "area", "areas_detail",
              "region", "scope", "netzero_year", "price_year", "matched_via",
              "n_price_rows", "price", "price_range", "price_first_seen",
              "raw_values", "source"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ============================================================================
# FIGURE
# ============================================================================

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": INK2,
        "axes.edgecolor": GRID, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 160,
    })


def _jitter(n, spread=0.22):
    if n <= 1:
        return [0.0]
    step = (2 * spread) / (n - 1)
    return [(-spread + i * step) for i in range(n)]


def _box(ax, x, values, width):
    v = sorted(values)
    q1, med, q3 = (statistics.quantiles(v, n=4)[0], statistics.median(v),
                   statistics.quantiles(v, n=4)[2])
    iqr = q3 - q1
    lo = max(min(v), q1 - 1.5 * iqr)
    hi = min(max(v), q3 + 1.5 * iqr)
    ax.add_patch(plt.Rectangle((x - width / 2, q1), width, iqr,
                               facecolor="#eef1f4", edgecolor=INK2, lw=1.2,
                               zorder=2))
    ax.plot([x - width / 2, x + width / 2], [med, med], color=INK, lw=2.2,
            zorder=3)
    ax.plot([x, x], [lo, q1], color=INK2, lw=1.0, zorder=2)
    ax.plot([x, x], [q3, hi], color=INK2, lw=1.0, zorder=2)


def figure_caveat(stats):
    return (
        "Exploratory. One system / wholesale electricity price per net-zero "
        "scenario at its last pathway year, already in real 2025 USD/MWh. "
        f"{stats['dropped_detail']} single-technology tariff / LCOE / feed-in "
        f"rows excluded. {stats['n']} scenarios from only {stats['n_papers']} "
        f"studies; net-zero years span "
        f"{stats['year_min']}–{stats['year_max']}."
    )


def fig_electricity_price(rows, stats, path):
    prices = [r["price"] for r in rows]

    def rc(r):
        return REGION_COLOR[r["region"]]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.0, 5.8), gridspec_kw={"width_ratios": [1, 2.7]},
        sharey=True)

    _box(ax1, 0, prices, width=0.5)
    for dy, r in zip(_jitter(len(rows)), rows):
        ax1.scatter([dy], [r["price"]], s=42, c=rc(r),
                    edgecolors="white", linewidths=0.7, alpha=0.95, zorder=5)
    ax1.set_xlim(-0.7, 0.7)
    ax1.set_xticks([0])
    ax1.set_xticklabels([f"all\nn={stats['n']}"])
    ax1.set_ylabel("electricity price at net zero  ·  real 2025 USD / MWh")
    ax1.axhline(stats["median"], color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax1.text(0.72, stats["median"], f"median ${stats['median']:,.0f}",
             fontsize=8, fontweight="bold", va="center", ha="left", color=INK)

    top = max(prices) * 1.18
    for i, r in enumerate(rows):
        ax2.plot([i, i], [0, r["price"]], color=GRID, lw=1.0, zorder=1)
        ax2.scatter([i], [r["price"]], s=44, c=rc(r),
                    edgecolors="white", linewidths=0.7, zorder=5)
        ax2.text(i, r["price"] + top * 0.02, f"${r['price']:,.0f}",
                 ha="center", fontsize=7.2, color=INK)
        area = r["area"] if len(r["area"]) <= 22 else r["area"][:21] + "…"
        ax2.text(i, -top * 0.03,
                 f"{area}\np{r['paper_id']} · {r['netzero_year']}",
                 ha="right", va="top", rotation=42, fontsize=6.8, color=INK2,
                 rotation_mode="anchor")
    ax2.set_xticks([])
    ax2.set_xlim(-0.9, len(rows) - 0.1)
    ax2.set_title("ranked low → high", fontsize=9, loc="left", color=MUTED,
                  pad=4)

    for ax in (ax1, ax2):
        ax.grid(axis="y", which="major", color=GRID, lw=0.8)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
    ax1.set_ylim(0, top)

    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=7,
                          mfc=REGION_COLOR[g], mec="white")
               for g in stats["regions"]]
    fig.legend(handles, stats["regions"], loc="lower center",
               ncol=min(len(stats["regions"]), 5), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), title="world region",
               title_fontsize=7.5)
    fig.suptitle(
        f"System electricity price in the year of net zero  ·  {stats['n']} "
        f"scenarios, {stats['n_papers']} papers",
        fontsize=11.5, x=0.03, ha="left", y=0.99)
    fig.text(0.5, -0.09, figure_caveat(stats), ha="center", va="top",
             fontsize=7.2, color=MUTED, wrap=True)

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# DOCUMENT
# ============================================================================

DOC_CSS = """
  :root{--ground:#f5f5f2;--panel:#efeeea;--panel-line:#dddcd4;--ink:#232427;
    --ink-soft:#4c4d4e;--muted:#7c7b73;--accent:#8f3a2e;--flag:#9a6a1c;
    --rule:#cfcec6;--code-bg:#ecebe5}
  @media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --ground:#16171a;--panel:#1d1e21;--panel-line:#2c2d31;--ink:#e9e7e1;
    --ink-soft:#c3c1b9;--muted:#8f8e85;--accent:#d98b5f;--flag:#d0a558;
    --rule:#2f3034;--code-bg:#232428}}
  :root[data-theme="dark"]{--ground:#16171a;--panel:#1d1e21;--panel-line:#2c2d31;
    --ink:#e9e7e1;--ink-soft:#c3c1b9;--muted:#8f8e85;--accent:#d98b5f;
    --flag:#d0a558;--rule:#2f3034;--code-bg:#232428}
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);
    font-family:"Newsreader",Georgia,serif;font-size:18px;line-height:1.62;
    font-variant-numeric:oldstyle-nums}
  .topbar{position:sticky;top:0;z-index:10;display:flex;
    justify-content:space-between;align-items:baseline;gap:1rem;
    padding:.7rem clamp(1rem,4vw,2.5rem);
    background:color-mix(in srgb,var(--ground) 88%,transparent);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);
    font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.74rem;
    letter-spacing:.04em;text-transform:uppercase}
  .topbar b{color:var(--accent);font-weight:600}.topbar span{color:var(--muted)}
  main{max-width:46rem;margin:0 auto;padding:0 clamp(1rem,4vw,2rem) 6rem}
  section{margin-top:4.5rem}
  .hero{margin-top:3.2rem}
  .kicker{font-family:"IBM Plex Mono",monospace;font-size:.76rem;
    letter-spacing:.14em;text-transform:uppercase;color:var(--accent);
    margin:0 0 1.1rem}
  h1{font-family:"IBM Plex Mono",monospace;font-weight:600;
    font-size:clamp(2rem,6vw,2.9rem);line-height:1.08;letter-spacing:-.01em;
    text-wrap:balance;margin:0 0 1rem}
  .lede{font-size:1.26rem;line-height:1.5;color:var(--ink-soft);
    max-width:35rem;margin:0 0 2.4rem}
  .lede em{font-style:italic;color:var(--ink)}
  h2{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:1.32rem;
    letter-spacing:-.005em;margin:0 0 1.1rem;padding-bottom:.5rem;
    border-bottom:1px solid var(--rule);text-wrap:balance}
  h3{font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:.95rem;
    letter-spacing:.02em;text-transform:uppercase;color:var(--ink-soft);
    margin:2.2rem 0 .8rem}
  p{margin:0 0 1.05rem;max-width:40rem}
  a{color:var(--accent)}
  strong{font-weight:600}
  ul{padding-left:1.2rem;margin:0 0 1.05rem;max-width:40rem}
  li{margin:.35rem 0}
  code{font-family:"IBM Plex Mono",monospace;font-size:.86em;
    background:var(--code-bg);padding:.06em .38em;border-radius:2px;
    white-space:nowrap}
  figure{margin:1.6rem 0 .6rem;padding:.9rem;background:var(--panel);
    border:1px solid var(--panel-line)}
  figure img{width:100%;height:auto;display:block}
  figcaption{font-family:"IBM Plex Mono",monospace;font-size:.74rem;
    color:var(--muted);margin-top:.7rem;line-height:1.5}
  .stat-row{display:flex;flex-wrap:wrap;gap:1.6rem 2.4rem;margin:1.6rem 0 1.2rem}
  .stat b{display:block;font-family:"IBM Plex Mono",monospace;font-weight:600;
    font-size:1.9rem;color:var(--accent);line-height:1.1;
    font-variant-numeric:tabular-nums}
  .stat span{font-family:"IBM Plex Mono",monospace;font-size:.72rem;
    text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
  .flow{display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch;
    margin:1.4rem 0 1.8rem;font-family:"IBM Plex Mono",monospace;font-size:.8rem}
  .flow .step{flex:1 1 8rem;background:var(--panel);
    border:1px solid var(--panel-line);padding:.7rem .8rem}
  .flow .step b{color:var(--accent);display:block;font-weight:600;
    margin-bottom:.2rem}
  .flow .arr{align-self:center;color:var(--muted)}
  .scroll{overflow-x:auto;margin:1.4rem 0 1.6rem}
  table{width:100%;border-collapse:collapse;font-family:"IBM Plex Mono",monospace;
    font-size:.8rem;font-variant-numeric:tabular-nums}
  th{text-align:left;font-weight:600;color:var(--ink-soft);
    text-transform:uppercase;letter-spacing:.04em;font-size:.7rem;
    padding:.5rem .8rem .5rem 0;border-bottom:2px solid var(--accent);
    white-space:nowrap}
  td{padding:.45rem .8rem .45rem 0;border-bottom:1px solid var(--rule);
    vertical-align:top}
  td.n,th.n{text-align:right;padding-right:1.2rem}
  tr:last-child td{border-bottom:none}
  .t-note{color:var(--muted);font-size:.78rem}
  .callout{background:var(--panel);border:1px solid var(--panel-line);
    border-left:3px solid var(--flag);padding:1rem 1.2rem;margin:1.4rem 0;
    font-size:.96rem}
  .callout ul{margin:0}
  footer{max-width:46rem;margin:5rem auto 0;padding:2rem clamp(1rem,4vw,2rem) 3rem;
    border-top:1px solid var(--rule);font-family:"IBM Plex Mono",monospace;
    font-size:.76rem;color:var(--muted)}
  footer code{background:none;color:var(--ink-soft)}
  @media(max-width:640px){body{font-size:16.5px}.lede{font-size:1.12rem}
    .stat b{font-size:1.5rem}}
  @media(prefers-reduced-motion:reduce){*{transition:none!important}}
"""


def build_doc(rows, stats, png_b64):
    s = stats
    esc = html.escape

    beccs_li = ""
    if s["beccs"]:
        b = s["beccs"]
        beccs_li = (
            f"<li><strong>Removals barely move the price.</strong> The one "
            f"paper with a paired comparison (<code>p{esc(b['paper'])}</code>) "
            f"lands at <strong>${round(b['with']):,}</strong>/MWh with CDR and "
            f"<strong>${round(b['without']):,}</strong> without &mdash; a "
            f"${round(abs(b['delta'])):,}/MWh difference.</li>")

    region_li = " · ".join(
        f"{g} ${v:,} (n={s['region_n'][g]})"
        for g, v in sorted(s["region_median"].items(), key=lambda kv: kv[1]))

    table_rows = "\n".join(
        f"          <tr><td class=\"n\">{r['price']:,.0f}</td>"
        f"<td>{esc(r['region'])}</td><td>{esc(r['area'])}</td>"
        f"<td>{esc(r['scenario'])}</td>"
        f"<td class=\"n\">{r['netzero_year']}</td>"
        f"<td>p{esc(r['paper_id'])}</td></tr>"
        for r in rows)

    return f"""<title>Electricity Price at Net Zero</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Newsreader:ital@1&display=swap">

<style>{DOC_CSS}</style>

<div class="topbar">
  <b>electricity&#8202;·&#8202;price</b>
  <span>scripts/08-stage3c-electricity-price.py</span>
</div>

<main>
  <div class="hero">
    <p class="kicker">Data extraction pipeline &nbsp;/&nbsp; analysis</p>
    <h1>Electricity Price at Net Zero</h1>
    <p class="lede">
      Of {s['flags_total']} scenarios flagged as reaching net zero,
      <em>{s['n']} also report a system electricity price</em> at the end of the
      pathway. In real 2025 dollars they run from
      <strong>${s['p_min']:,}</strong> to <strong>${s['p_max']:,}</strong> per
      MWh &mdash; a {s['spread']}&times; spread across {s['n_papers']} studies.
    </p>
  </div>

  <figure>
    <img src="data:image/png;base64,{png_b64}" alt="Box plot and ranked strip plot of {s['n']} net-zero-year electricity prices, real 2025 USD per MWh, from ${s['p_min']:,} to ${s['p_max']:,}, median ${s['median']:,}.">
    <figcaption>
      figures/electricity-price-at-netzero.png &mdash; left: the distribution
      (box = IQR, line = median). right: every scenario, ranked, labelled by
      area &middot; paper &middot; net-zero year. colour = world region.
    </figcaption>
  </figure>

  <section>
    <h2>What it shows</h2>
    <div class="stat-row">
      <div class="stat"><b>${s['median']:,}</b><span>median /MWh</span></div>
      <div class="stat"><b>${s['q1']:,}&ndash;{s['q3']:,}</b><span>interquartile range</span></div>
      <div class="stat"><b>${s['p_min']:,}&ndash;{s['p_max']:,}</b><span>full range</span></div>
      <div class="stat"><b>{s['n']} / {s['n_papers']}</b><span>scenarios / papers</span></div>
    </div>
    <ul>
      <li><strong>A {s['spread']}&times; spread.</strong> The cheapest net-zero
        electricity price (<code>p{s['low_paper']}</code>, {esc(s['low_area'])},
        <strong>${s['low_price']:,}</strong>/MWh) and the most expensive
        (<code>p{s['high_paper']}</code>, {esc(s['high_area'])},
        <strong>${s['high_price']:,}</strong>) differ by more than the whole
        range of today's wholesale markets &mdash; this is a modelling
        assumption gap, not a physical one.</li>
      {beccs_li}
      <li><strong>Median by world region:</strong> {region_li}. Each region is
        one or two studies, so region and study effect are not separable here
        &mdash; within a study the scenarios sit close together, the gap is
        <em>between</em> studies / regions.</li>
      <li>One study (<code>p{s['dom_paper']}</code>, {esc(s['dom_area'])})
        contributes {s['dom_n']} of the {s['n']} scenarios ({s['dom_pct']}%),
        all at the high end (${s['dom_min']:,}&ndash;{s['dom_max']:,});
        it pulls the median up.</li>
    </ul>
  </section>

  <section>
    <h2>Method</h2>
    <div class="flow">
      <div class="step"><b>1</b> net-zero filter</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>2</b> electricity-price rows</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>3</b> drop single-tech tariffs</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>4</b> price at the neutrality year</div>
    </div>
    <ul>
      <li><strong>Net-zero membership</strong> comes from
        <code>netzero-flags.csv</code> (<code>is_netzero&nbsp;==&nbsp;True</code>)
        &mdash; the same filter as the carbon-price and sector-balance
        figures, matched on canonical scenario / area keys.</li>
      <li><strong>Electricity-price rows</strong> are
        <code>harmonized-costs.csv</code> rows with
        <code>Cost_Type = "Electricity price"</code> and a value already in
        <code>USD2025/MWh</code>. Rows whose <code>Cost_Type_Detail</code> names
        a single-technology on-grid tariff, a feed-in tariff, a PPA or an LCOE
        quote &mdash; and a mislabelled staple-food series in one paper &mdash;
        are dropped ({s['dropped_detail']} rows). A price whose scenario field
        is a catch-all is applied to every flagged scenario of that paper that
        has none of its own ({s['applied_catchall']} rows here).</li>
      <li><strong>End year</strong> = the scenario&rsquo;s neutrality year from
        <code>netzero-flags.csv</code>; the price is read at that year or the
        nearest earlier one the scenario reports (<code>price_year</code> in the
        CSV). One value per <code>(paper,&nbsp;scenario)</code> &mdash; the
        median across every area and row at that year, so a study that quotes
        the same price for four grid regions counts once.</li>
      <li><strong>World region</strong> is keyword-matched from the
        <code>Area</code> string (country / sub-national unit &rarr; region);
        a scenario spanning regions is filed as
        <em>Global / multi-region</em>.</li>
    </ul>
  </section>

  <section>
    <h2>The {s['n']} scenarios</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr><th class="n">USD2025 /MWh</th><th>Region</th><th>Area</th>
            <th>Scenario</th><th class="n">Net-zero yr</th><th>Paper</th></tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
      </table>
    </div>
    <p class="t-note">Full table with the first-year price, titles and DOIs:
      <code>outputs/analysis/electricity-price-at-netzero.csv</code>.</p>
  </section>

  <section>
    <h2>Caveats</h2>
    <div class="callout">
      <ul>
        <li><strong>n = {s['n']}</strong> from only {s['n_papers']} papers, and
          one study (<code>p{s['dom_paper']}</code>) is {s['dom_pct']}% of it.
          Treat this as a range of modelling assumptions, not a distribution.</li>
        <li>&ldquo;System electricity price&rdquo; is not defined identically
          across studies &mdash; wholesale, retail, LCOE-of-the-system and
          shadow price of the electricity balance are pooled here.</li>
        <li>Retail vs wholesale is the likely driver of the high cluster; the
          extraction does not always record which.</li>
        <li>Net-zero years span
          <strong>{s['year_min']}&ndash;{s['year_max']}</strong>; a later date
          is not obviously cheaper or dearer in this sample.</li>
        <li>The net-zero flag has a threshold
          (<code>--netzero-fraction</code>, default 0.05); the real bottleneck
          is that <strong>most net-zero scenarios report no electricity
          price at all</strong>.</li>
      </ul>
    </div>
  </section>
</main>

<footer>
  reproduce &nbsp;&rarr;&nbsp; <code>python3 scripts/08-stage3c-electricity-price.py</code><br>
  inputs: <code>harmonized-costs.csv</code>, <code>netzero-flags.csv</code>
  &nbsp;&middot;&nbsp; outputs: <code>figures/electricity-price-at-netzero.png</code>,
  <code>outputs/analysis/electricity-price-at-netzero.csv</code>
</footer>
"""


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("costs", nargs="?", default=str(DEFAULT_COSTS))
    ap.add_argument("--flags", default=str(DEFAULT_FLAGS))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--figures-dir", default=str(DEFAULT_FIGDIR))
    ap.add_argument("--doc", default=str(DEFAULT_DOC))
    args = ap.parse_args()

    flags = [r for r in load_csv(Path(args.flags))
             if r.get("is_netzero") == "True"]
    rows, meta = build(load_csv(Path(args.costs)), flags)
    if not rows:
        raise SystemExit("no net-zero scenario reports a system electricity price")

    stats = compute_stats(rows, meta)
    write_csv(Path(args.outdir) / "electricity-price-at-netzero.csv", rows)

    fig_path = Path(args.figures_dir) / "electricity-price-at-netzero.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _style()
    fig_electricity_price(rows, stats, fig_path)

    png_b64 = base64.b64encode(fig_path.read_bytes()).decode()
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_doc(rows, stats, png_b64), encoding="utf-8")

    print(f"net-zero scenarios flagged   : {stats['flags_total']}")
    print(f"  single-tech rows dropped    : {stats['dropped_detail']}")
    print(f"  paper-wide prices applied   : {stats['applied_catchall']}")
    print(f"  with a system price         : {stats['n']}  "
          f"({stats['n_papers']} papers)")
    print(f"  USD2025/MWh : min {stats['p_min']:,}  Q1 {stats['q1']:,}  "
          f"median {stats['median']:,}  Q3 {stats['q3']:,}  "
          f"max {stats['p_max']:,}  (spread {stats['spread']}x)")
    print(f"\nfigure -> {fig_path}\ndoc    -> {doc_path}")


if __name__ == "__main__":
    main()
