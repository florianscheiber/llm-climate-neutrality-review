# scripts/10-stage3c-macro-economic-cost.py  (stage 3c -- exploratory analysis)
#
# The economy-wide cost of the transition, as the reviewed studies report it.
# Three panels, each a thin slice of harmonized-costs.csv:
#
#   A  GDP impact              % change in GDP versus a reference / BAU run,
#                              at the last reported year. Sign harmonised so
#                              a negative value = GDP below the reference.
#
#   B  Total system cost       % change in total energy / power system cost
#                              versus a reference scenario. "reduction" and
#                              "increase" phrasings are signed accordingly;
#                              ratio values (1.014) are converted to +1.4 %.
#
#   C  Investment need         economy- or energy-system-wide capital
#                              investment, annualised to USD billion / year
#                              (a cumulative figure is divided by the length
#                              of its accumulation window). LOG axis.
#
# This is the thinnest analysis in the pipeline -- each panel is a handful
# of studies and the metrics are only loosely comparable (different
# reference scenarios, accounting boundaries, geographies). The figure is
# built to make that explicit: every point is labelled and n is stated.
#
# Every number in the figure and the doc is computed here.
#
# INPUT   outputs/output-stage3c/harmonized-costs.csv
#         outputs/output-stage3c/netzero-flags.csv
# OUTPUT  outputs/analysis/macro-economic-cost.csv
#         figures/macro-economic-cost.png
#         docs/macro-economic-cost.html
#
#   python3 scripts/10-stage3c-macro-economic-cost.py   (run from the repo root)

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
from _pipeline_common import canon_scenario, resolve_scenario   # noqa: E402

INK, INK2, MUTED, GRID = "#1a1a1a", "#55534e", "#8a8a84", "#dedcd5"

# world region -> colourblind-safe hue (shared with 08 / 09)
REGION_ORDER = [
    "North America", "Latin America", "Europe", "Africa", "Middle East",
    "China", "India", "Rest of Asia-Pacific", "Global / multi-region",
    "Not reported",
]
REGION_COLOR = {
    "North America": "#0072b2", "Latin America": "#b07d2b",
    "Europe": "#56b4e9", "Africa": "#cc79a7", "Middle East": "#009e73",
    "China": "#d55e00", "India": "#e69f00",
    "Rest of Asia-Pacific": "#7a5195", "Global / multi-region": "#8a8a84",
    "Not reported": "#c2c0b8",
}
_REGION_RULES = [
    ("Global / multi-region",
     r"\b(global|world(wide)?|multi-?region|gcam regions|all regions|"
     r"model regions|whole model|aggregate)\b"),
    ("China",
     r"\bchina\b|\bprc\b|chinese|nanning|shanxi|guangdong|guangxi|beijing|"
     r"shanghai|shenzhen|sichuan|inner mongolia|yangtze|xian|hohhot|"
     r"shenyang|yichang|chengdu|^(ea|na|sa|wa|nc|sc|ec|wc)$"),
    ("India", r"\bindia\b|indian\b|\bdelhi\b"),
    ("Middle East",
     r"\bsaudi|\buae\b|emirates|\bqatar|\bkuwait|\boman\b|\bbahrain|"
     r"\biran\b|\biraq\b|israel|\bjordan|middle east|\bgcc\b|persian gulf"),
    ("Africa",
     r"\bafrica|nigeria|\begypt|kenya|\bghana|ethiopia|morocco|tanzania|"
     r"\bsub-?saharan"),
    ("Europe",
     r"\b(europe|european union|eu\+?|euro area|portugal|spain|france|"
     r"germany|german|italy|poland|polish|voivodeship|ma[a-z]*opolska|"
     r"switzerland|swiss|austria|belgium|netherlands|dutch|norway|sweden|"
     r"finland|finnish|denmark|ireland|united kingdom|\buk\b|britain|"
     r"scotland|wales|greece|czech|slovak|hungary|romania|bulgaria|"
     r"latvia|lithuania|estonia|baltic|croatia|serbia|slovenia|iceland)\b"),
    ("North America",
     r"\b(united states|\bu\.?s\.?a?\.?\b|america\b|canada|canadian|quebec|"
     r"mexico|california|texas|new york|bay area|western us|pjm|ercot)\b"),
    ("Latin America",
     r"\b(brazil|brazilian|chile|chilean|argentina|colombia|peru|uruguay|"
     r"latin america|south america)\b"),
    ("Rest of Asia-Pacific",
     r"\b(japan|japanese|korea|korean|australia|australian|new zealand|"
     r"thailand|thai|nakhon|vietnam|indonesia|malaysia|philippines|"
     r"singapore|taiwan|pakistan|asean|asia|pacific|oceania)\b"),
]


def world_region(area):
    a = (area or "").strip()
    if not a or a.lower() in ("not reported", "n/a", "unspecified", "unknown",
                              "none"):
        return "Not reported"
    for name, pat in _REGION_RULES:
        if re.search(pat, a, re.IGNORECASE):
            return name
    return "Not reported"


DEFAULT_COSTS = Path("outputs/output-stage3c/harmonized-costs.csv")
DEFAULT_FLAGS = Path("outputs/output-stage3c/netzero-flags.csv")
DEFAULT_OUTDIR = Path("outputs/analysis")
DEFAULT_FIGDIR = Path("figures")
DEFAULT_DOC = Path("docs/macro-economic-cost.html")


# ============================================================================
# HELPERS
# ============================================================================

def to_float(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def year_span(raw):
    """(first, last) years from '2016-2050' / '2020–2050' / '2050'."""
    ys = [int(y) for y in re.findall(r"(?:19|20)\d{2}", raw or "")]
    if not ys:
        return None, None
    return min(ys), max(ys)


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def g(r, k):
    return (r.get(k) or "").strip()


def short(s, n_):
    s = s.strip()
    return s if len(s) <= n_ else s[:n_ - 1] + "…"


# ============================================================================
# PANEL A -- GDP impact (% vs reference)
# ============================================================================

_GDP_KEEP = re.compile(r"gdp (loss|change|impact)|real gdp|gross domestic",
                       re.I)
_GDP_DROP = re.compile(r"growth rate|per capita|projection|"
                       r"additional (total|industry|services|agriculture)",
                       re.I)


def panel_a(costs):
    out = {}
    for r in costs:
        if g(r, "Cost_Type") != "(Change in) GDP":
            continue
        unit = g(r, "Value_Unit").lower()
        if "%" not in unit and "percentage point" not in unit:
            continue
        detail = g(r, "Cost_Type_Detail")
        if not _GDP_KEEP.search(detail) or _GDP_DROP.search(detail):
            continue
        v = to_float(g(r, "Value"))
        if v is None:
            continue
        # harmonise sign: a "loss" quoted as a positive magnitude -> negative
        if re.search(r"\bloss\b", detail, re.I) and v > 0:
            v = -v
        _, last = year_span(g(r, "Year"))
        key = (g(r, "paper_id"), g(r, "Scenario_Name"))
        cur = out.get(key)
        if cur is None or (last or 0) >= (cur["year"] or 0):
            out[key] = {"paper_id": g(r, "paper_id"),
                        "title": g(r, "Title"), "doi": g(r, "DOI"),
                        "scenario": g(r, "Scenario_Name"),
                        "area": g(r, "Area"),
                        "region": world_region(g(r, "Area")),
                        "year": last, "value": round(v, 2),
                        "detail": detail}
    return sorted(out.values(), key=lambda d: d["value"])


# ============================================================================
# PANEL B -- total system cost (% vs reference)
# ============================================================================

_SYS_KEEP = re.compile(
    r"system cost|discounted cost|energy-?production cost|"
    r"energy-?system cost|levelized system|total (energy|power) cost", re.I)
_SYS_CTX = re.compile(r"relative|reference|change|difference|increase|"
                      r"reduction|saving|versus|compared", re.I)
_SYS_DROP = re.compile(r"discount rate|share in system cost|"
                       r"share of (the )?(overall|system)|decomposition", re.I)


def panel_b(costs):
    out = {}
    for r in costs:
        if g(r, "Cost_Type") not in ("System cost (total)", "Other"):
            continue
        detail = g(r, "Cost_Type_Detail")
        if not _SYS_KEEP.search(detail) or not _SYS_CTX.search(detail):
            continue
        if _SYS_DROP.search(detail):
            continue
        unit = g(r, "Value_Unit").lower()
        v = to_float(g(r, "Value"))
        if v is None:
            continue
        if "%" in unit or "percent" in unit:
            pct = v
        elif re.search(r"relative|ratio|index", unit):
            pct = (v - 1.0) * 100.0        # 1.014 -> +1.4
        else:
            continue                        # absolute currency -> skip here
        if re.search(r"reduction|saving", detail, re.I) and pct > 0:
            pct = -pct
        _, last = year_span(g(r, "Year"))
        key = (g(r, "paper_id"), g(r, "Scenario_Name"))
        cur = out.get(key)
        # keep the largest-magnitude change per scenario (the headline number)
        if cur is None or abs(pct) > abs(cur["value"]):
            out[key] = {"paper_id": g(r, "paper_id"),
                        "title": g(r, "Title"), "doi": g(r, "DOI"),
                        "scenario": g(r, "Scenario_Name"),
                        "area": g(r, "Area"),
                        "region": world_region(g(r, "Area")),
                        "year": last, "value": round(pct, 2),
                        "detail": detail}
    return sorted(out.values(), key=lambda d: d["value"])


# ============================================================================
# PANEL C -- investment need (annualised, USD bn / year)
# ============================================================================

_INV_KEEP = re.compile(
    r"total investment|cumulative capital investment|investment across|"
    r"investment need|capital committed for clean|investment required|"
    r"combined investment|annual (capital )?investment|investment surge|"
    r"total electricity-?system investment|power-?sector investment|"
    r"cumulative capital investment", re.I)
_INV_DROP = re.compile(
    r"\b(wind|solar|pv|nuclear|hydro|geotherm|ccs|beccs|daccs?|\bdac\b|"
    r"battery|storage|electroly|transmission|distribution|cement|ammonia|"
    r"desalination|fisher|biofuel|gas with carbon)\b|per unit|per cumulative|"
    r"building renovation|hydrogen infrastructure|efficiency investment|"
    r"local stakeholders", re.I)


def panel_c(costs):
    out = {}
    for r in costs:
        if g(r, "Cost_Type") != "Investment cost (total)":
            continue
        detail = g(r, "Cost_Type_Detail")
        if not _INV_KEEP.search(detail) or _INV_DROP.search(detail):
            continue
        unit = g(r, "Value_std_Unit")
        v = to_float(g(r, "Value_std"))
        if v is None or v <= 0:
            continue
        first, last = year_span(g(r, "Year"))
        if unit == "USD2025/yr":
            annual, basis = v, "reported annual"
        elif unit == "USD2025":
            # a single stated year is the accumulation END; assume a 2020 start
            start = first if (first and last and first != last) else 2020
            end = last or 2050
            span = max(end - start, 1)
            annual, basis = v / span, f"cumulative / {span} yr"
        else:
            continue
        bn = annual / 1e9
        key = (g(r, "paper_id"), g(r, "Scenario_Name"))
        cur = out.get(key)
        # prefer a directly-reported annual figure; otherwise the widest-scope
        # (largest) cumulative-derived one
        better = (
            cur is None
            or (basis == "reported annual" and cur["basis"] != "reported annual")
            or (basis == cur["basis"] and bn > cur["value"])
            or (cur["basis"] != "reported annual" and bn > cur["value"]
                and basis != "reported annual")
        )
        if better:
            out[key] = {"paper_id": g(r, "paper_id"),
                        "title": g(r, "Title"), "doi": g(r, "DOI"),
                        "scenario": g(r, "Scenario_Name"),
                        "area": g(r, "Area"),
                        "region": world_region(g(r, "Area")),
                        "year": last, "value": round(bn, 2),
                        "basis": basis, "detail": detail}
    return sorted(out.values(), key=lambda d: d["value"])


# ============================================================================
# BUILD
# ============================================================================

def build(costs, flags):
    # {paper_id: {scenario_canon: True}} for flagged net-zero scenarios
    nz = {}
    for f in flags:
        if f.get("is_netzero") != "True":
            continue
        sc = f.get("scenario_canon") or canon_scenario(f.get("Scenario_Name"))
        nz.setdefault(f["paper_id"], {})[sc] = True
    panels = {"gdp": panel_a(costs), "system": panel_b(costs),
              "investment": panel_c(costs)}
    for name, rows in panels.items():
        for d in rows:
            d["panel"] = name
            per_paper = nz.get(d["paper_id"], {})
            sc = canon_scenario(d["scenario"])
            d["is_netzero"] = bool(
                per_paper.get(sc) or resolve_scenario(sc, per_paper))
    return panels


def per_paper(rows):
    """Collapse a panel's scenario rows to one entry per paper: median value,
    min/max range, region, net-zero status, an example scenario label."""
    by = {}
    for r in rows:
        by.setdefault(r["paper_id"], []).append(r)
    out = []
    for pid, rs in by.items():
        vals = sorted(x["value"] for x in rs)
        lo_r = min(rs, key=lambda x: x["value"])
        hi_r = max(rs, key=lambda x: x["value"])
        out.append({
            "paper_id": pid,
            "title": rs[0]["title"], "doi": rs[0]["doi"],
            "region": rs[0]["region"],
            "area": rs[0]["area"],
            "n_scen": len(rs),
            "is_netzero": any(x["is_netzero"] for x in rs),
            "median": round(statistics.median(vals), 2),
            "min": vals[0], "max": vals[-1],
            "lo_scen": lo_r["scenario"], "hi_scen": hi_r["scenario"],
        })
    return sorted(out, key=lambda d: d["median"])


def compute_stats(panels):
    def block(rows):
        pp = per_paper(rows)
        vals = [r["value"] for r in rows]
        papers = sorted({r["paper_id"] for r in rows}, key=int)
        regions = [g for g in REGION_ORDER
                   if any(r["region"] == g for r in rows)]
        return {
            "n": len(rows), "n_papers": len(papers), "papers": papers,
            "regions": regions, "per_paper": pp,
            "min": round(min(vals), 2), "max": round(max(vals), 2),
            "median": round(statistics.median(vals), 2),
            "paper_median_lo": pp[0]["median"], "paper_median_hi": pp[-1]["median"],
            "n_netzero": sum(1 for r in rows if r["is_netzero"]),
        }
    return {name: block(rows) for name, rows in panels.items()}


def write_csv(path, panels):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["panel", "paper_id", "title", "doi", "scenario", "area",
              "region", "is_netzero", "year", "value", "basis", "detail"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";",
                           extrasaction="ignore")
        w.writeheader()
        for name in ("gdp", "system", "investment"):
            for d in panels[name]:
                w.writerow({**d, "value": d["value"]})


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


def _paper_strip(ax, pp, stats, xlabel, log=False, fmt="{:+.1f}%", zero=True):
    """One row per paper: a min-max range line + a median marker, labelled."""
    n = len(pp)
    for y, r in enumerate(pp):
        col = REGION_COLOR[r["region"]]
        if r["max"] > r["min"]:
            ax.plot([r["min"], r["max"]], [y, y], color=col, lw=2.6,
                    solid_capstyle="round", alpha=0.55, zorder=3)
        ax.scatter([r["median"]], [y], s=64, c=col, edgecolors="white",
                   linewidths=0.9, marker="o" if r["is_netzero"] else "D",
                   zorder=5)
        tag = f" ({r['n_scen']} scen)" if r["n_scen"] > 1 else ""
        ax.annotate(f"p{r['paper_id']} · {short(r['area'], 18)}{tag}",
                    (r["max"], y), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=6.8, color=INK2)
        rng = (fmt.format(r["min"]) if r["max"] == r["min"]
               else f"{fmt.format(r['min'])} … {fmt.format(r['max'])}")
        ax.annotate(rng, (r["min"], y), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=6.6, color=INK, fontweight="bold")
    if zero and not log:
        ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.axvline(stats["median"], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if log:
        ax.set_xscale("log")
        lo = min(r["min"] for r in pp)
        hi = max(r["max"] for r in pp)
        ax.set_xlim(lo * 0.35, hi * 3.2)
    else:
        lo = min(r["min"] for r in pp)
        hi = max(r["max"] for r in pp)
        pad = (hi - lo) * 0.28 + 1
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)


def fig_macro(panels, stats, path):
    a, b, c = stats["gdp"], stats["system"], stats["investment"]
    heights = [len(a["per_paper"]), len(b["per_paper"]), len(c["per_paper"])]
    fig, axes = plt.subplots(
        3, 1, figsize=(9.4, sum(0.42 * h + 1.7 for h in heights)),
        gridspec_kw={"height_ratios": [h + 3 for h in heights]})

    axes[0].set_title(
        f"A  ·  GDP versus a reference run  ·  {a['n_papers']} papers, "
        f"{a['n']} scenarios",
        loc="left", fontsize=10, color=INK, pad=12)
    _paper_strip(axes[0], a["per_paper"], a,
                 "change in GDP at the last reported year  ·  % vs reference / BAU")

    axes[1].set_title(
        f"B  ·  Total system cost versus a reference scenario  ·  "
        f"{b['n_papers']} papers, {b['n']} scenarios",
        loc="left", fontsize=10, color=INK, pad=12)
    _paper_strip(axes[1], b["per_paper"], b,
                 "change in total energy / power system cost  ·  % vs reference")

    axes[2].set_title(
        f"C  ·  Investment need, annualised  ·  {c['n_papers']} papers, "
        f"{c['n']} scenarios",
        loc="left", fontsize=10, color=INK, pad=12)
    _paper_strip(axes[2], c["per_paper"], c,
                 "capital investment  ·  USD billion / year  (log scale)",
                 log=True, fmt="{:,.0f}B", zero=False)

    # legend: region + netzero marker
    regs = [g for g in REGION_ORDER
            if any(r["region"] == g for p in panels.values() for r in p)]
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=7,
                          mfc=REGION_COLOR[g], mec="white") for g in regs]
    handles += [
        plt.Line2D([0], [0], marker="o", ls="", ms=7, mfc=MUTED, mec="white"),
        plt.Line2D([0], [0], marker="D", ls="", ms=6, mfc=MUTED, mec="white"),
        plt.Line2D([0], [0], color=MUTED, lw=1.0, ls=(0, (4, 3))),
    ]
    labels = regs + ["scenario flagged net-zero", "not flagged", "panel median"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.5, -0.055),
               title="colour = world region", title_fontsize=7.6)

    fig.suptitle("The economy-wide cost of the transition, as reported",
                 fontsize=12.5, x=0.02, ha="left", y=0.995)
    fig.text(0.5, -0.11,
             "Exploratory and thin. Each panel is a handful of studies and "
             "the metrics are only loosely comparable: reference scenarios, "
             "accounting boundaries and geographies differ. In C a cumulative "
             "figure is divided by its accumulation window; whole-economy and "
             "power-only totals are pooled.",
             ha="center", va="top", fontsize=7.2, color=MUTED, wrap=True)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=3.4)
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
    font-size:1.7rem;color:var(--accent);line-height:1.1;
    font-variant-numeric:tabular-nums}
  .stat span{font-family:"IBM Plex Mono",monospace;font-size:.72rem;
    text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
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
    .stat b{font-size:1.4rem}}
  @media(prefers-reduced-motion:reduce){*{transition:none!important}}
"""


def _table(rows, valfmt):
    esc = html.escape
    return "\n".join(
        f"          <tr><td>p{esc(r['paper_id'])}</td>"
        f"<td>{esc(r['region'])}</td>"
        f"<td>{esc(short(r['scenario'], 34))}</td>"
        f"<td class=\"n\">{r['year'] or '&mdash;'}</td>"
        f"<td class=\"n\">{valfmt(r['value'])}</td>"
        f"<td>{esc(short(r['detail'], 44))}</td></tr>"
        for r in rows)


def build_doc(panels, stats, png_b64):
    s = stats
    esc = html.escape
    gp, sp, ip = stats["gdp"], stats["system"], stats["investment"]

    def rng(b, f):
        return f"{f(b['min'])} to {f(b['max'])}"

    pf = lambda v: f"{v:+.1f}%"
    bf = lambda v: f"${v:,.0f}B/yr"

    return f"""<title>Macro-Economic Cost</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Newsreader:ital@1&display=swap">

<style>{DOC_CSS}</style>

<div class="topbar">
  <b>macro-economic&#8202;·&#8202;cost</b>
  <span>scripts/10-stage3c-macro-economic-cost.py</span>
</div>

<main>
  <div class="hero">
    <p class="kicker">Data extraction pipeline &nbsp;/&nbsp; analysis</p>
    <h1>Macro-Economic Cost</h1>
    <p class="lede">
      Besides a carbon price, how do the reviewed studies express the
      <em>economy-wide cost</em> of getting to net zero? Three metrics turn
      up &mdash; GDP impact, total system cost, investment need &mdash; and
      <em>each is reported by only a handful of papers</em>, in forms that
      barely compare.
    </p>
  </div>

  <figure>
    <img src="data:image/png;base64,{png_b64}" alt="Three stacked strip plots: GDP change vs reference ({gp['n']} scenarios), total system cost change vs reference ({sp['n']} scenarios), and annualised investment need on a log axis ({ip['n']} scenarios), coloured by world region.">
    <figcaption>
      figures/macro-economic-cost.png &mdash; three panels, each one thin
      slice of <code>harmonized-costs.csv</code>. dot = one scenario, colour
      = world region, circle = net-zero-flagged, diamond = not flagged.
      dashed line = panel median.
    </figcaption>
  </figure>

  <section>
    <h2>Panel A &mdash; GDP impact</h2>
    <div class="stat-row">
      <div class="stat"><b>{gp['n']} / {gp['n_papers']}</b><span>scenarios / papers</span></div>
      <div class="stat"><b>{rng(gp, pf)}</b><span>range vs reference</span></div>
      <div class="stat"><b>{gp['median']:+.1f}%</b><span>median</span></div>
    </div>
    <p>
      Only <strong>{gp['n_papers']} studies</strong> report GDP against a
      reference run. Signs are harmonised so a negative number means GDP sits
      below the reference. The spread &mdash;
      <strong>{rng(gp, pf)}</strong> &mdash; is mostly the difference between
      an R&amp;D-optimistic framing (small or positive) and a
      constrained-removals framing (several percent of GDP).
    </p>
    <div class="scroll"><table>
      <thead><tr><th>Paper</th><th>Region</th><th>Scenario</th>
        <th class="n">Year</th><th class="n">&Delta; GDP</th><th>As reported</th></tr></thead>
      <tbody>
{_table(panels['gdp'], pf)}
      </tbody></table></div>
  </section>

  <section>
    <h2>Panel B &mdash; total system cost</h2>
    <div class="stat-row">
      <div class="stat"><b>{sp['n']} / {sp['n_papers']}</b><span>scenarios / papers</span></div>
      <div class="stat"><b>{rng(sp, pf)}</b><span>range vs reference</span></div>
      <div class="stat"><b>{sp['median']:+.1f}%</b><span>median</span></div>
    </div>
    <p>
      The percentage change in a study's total energy- or power-system cost
      between a decarbonisation scenario and its own reference. The reference
      is not the same thing across studies &mdash; a BAU run, an
      unconstrained least-cost design, another net-zero variant &mdash; so
      the sign is more meaningful than the level. A dedicated-CCS or
      constrained-technology scenario is where the large positive values come
      from; several studies find the transition scenario is <em>cheaper</em>
      than their reference.
    </p>
    <div class="scroll"><table>
      <thead><tr><th>Paper</th><th>Region</th><th>Scenario</th>
        <th class="n">Year</th><th class="n">&Delta; cost</th><th>As reported</th></tr></thead>
      <tbody>
{_table(panels['system'], pf)}
      </tbody></table></div>
  </section>

  <section>
    <h2>Panel C &mdash; investment need</h2>
    <div class="stat-row">
      <div class="stat"><b>{ip['n']} / {ip['n_papers']}</b><span>scenarios / papers</span></div>
      <div class="stat"><b>{rng(ip, bf)}</b><span>annualised range</span></div>
      <div class="stat"><b>${ip['median']:,.0f}B/yr</b><span>median</span></div>
    </div>
    <p>
      Economy- or energy-system-wide capital investment, annualised to
      billion USD per year (a cumulative figure divided by its accumulation
      window). The <strong>{rng(ip, bf)}</strong> range is almost entirely
      geography and accounting boundary: a global whole-energy figure next to
      a single-country power-sector one. This panel shows there is
      <strong>no comparable number</strong>, not a distribution.
    </p>
    <div class="scroll"><table>
      <thead><tr><th>Paper</th><th>Region</th><th>Scenario</th>
        <th class="n">Year</th><th class="n">$B / yr</th><th>Basis</th></tr></thead>
      <tbody>
{_table(panels['investment'], bf)}
      </tbody></table></div>
  </section>

  <section>
    <h2>Method</h2>
    <ul>
      <li><strong>Panel A</strong> &mdash; <code>harmonized-costs.csv</code>
        rows with <code>Cost_Type = "(Change in) GDP"</code> and a
        <code>%</code> value whose label names a GDP loss / change / impact
        against a reference (growth rates, per-capita and absolute GDP levels
        excluded). A "loss" quoted as a positive magnitude is negated. One
        value per scenario, at the last reported year.</li>
      <li><strong>Panel B</strong> &mdash; rows in
        <code>System cost (total)</code> or <code>Other</code> whose label
        names a system / discounted / energy-production cost <em>and</em> a
        relative comparison. <code>%</code> is used directly; a ratio
        (<code>1.014</code>) becomes <code>+1.4%</code>; "reduction" /
        "saving" phrasings are signed negative. One value per scenario &mdash;
        the largest-magnitude change.</li>
      <li><strong>Panel C</strong> &mdash; <code>Investment cost (total)</code>
        rows for an economy- or energy-system-wide total (single-technology
        and single-sector lines dropped). <code>USD2025/yr</code> is used as
        is; <code>USD2025</code> is treated as cumulative and divided by the
        span of its <code>Year</code> range (start defaults to 2020). One
        value per scenario &mdash; the widest-scope figure.</li>
      <li>Net-zero flag from <code>netzero-flags.csv</code>; many of these
        scenarios are carbon-neutral by construction (the metric is "relative
        to BAU") but were not flagged because no emission time series was
        extracted, so the flag is shown, not used as a filter.</li>
    </ul>
  </section>

  <section>
    <h2>Caveats</h2>
    <div class="callout">
      <ul>
        <li>This is the <strong>thinnest analysis in the pipeline</strong>
          &mdash; {gp['n_papers']}, {sp['n_papers']} and {ip['n_papers']}
          papers per panel. Do not read a distribution into any of it.</li>
        <li>The three panels are <strong>not the same scenarios</strong> and
          mostly not the same papers.</li>
        <li>"Reference scenario" means something different in every study
          (Panels A, B).</li>
        <li>Panel C pools whole-economy and power-only investment, and a
          cumulative-to-annual conversion assumes a flat profile.</li>
        <li>GDP-loss estimates are model-structure artefacts as much as
          findings &mdash; CGE vs partial-equilibrium framing dominates.</li>
      </ul>
    </div>
  </section>
</main>

<footer>
  reproduce &nbsp;&rarr;&nbsp; <code>python3 scripts/10-stage3c-macro-economic-cost.py</code><br>
  inputs: <code>harmonized-costs.csv</code>, <code>netzero-flags.csv</code>
  &nbsp;&middot;&nbsp; outputs: <code>figures/macro-economic-cost.png</code>,
  <code>outputs/analysis/macro-economic-cost.csv</code>
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

    costs = load_csv(Path(args.costs))
    flags = load_csv(Path(args.flags))
    panels = build(costs, flags)
    if not any(panels.values()):
        raise SystemExit("no macro-economic cost rows found")

    stats = compute_stats(panels)
    write_csv(Path(args.outdir) / "macro-economic-cost.csv", panels)

    fig_path = Path(args.figures_dir) / "macro-economic-cost.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _style()
    fig_macro(panels, stats, fig_path)

    png_b64 = base64.b64encode(fig_path.read_bytes()).decode()
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_doc(panels, stats, png_b64), encoding="utf-8")

    for name in ("gdp", "system", "investment"):
        st = stats[name]
        print(f"{name:11}: {st['n']:2} scenarios, {st['n_papers']} papers, "
              f"range {st['min']} .. {st['max']}  (median {st['median']})")
    print(f"\nfigure -> {fig_path}\ndoc    -> {doc_path}")


if __name__ == "__main__":
    main()
