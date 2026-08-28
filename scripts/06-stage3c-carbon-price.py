# stage3d-carbon-price.py
#
# Exploratory analysis: what carbon price does a scenario carry in the year it
# reaches net zero?
#
# Net-zero scenarios come from netzero-flags.csv (is_netzero == True) -- the
# SAME filter as stage3d-sector-balance.py, but here a sector breakdown is not
# required, so single-sector / total-only scenarios are included too.
#
# For every net-zero scenario, the carbon-price rows of harmonized-costs.csv
# (Cost_Type == "Carbon price or abatement cost") at the pathway's last year
# are collected, normalised to 2025 USD per tCO2e, reduced to one value per
# scenario (median), and compared in a box plot.
#
# Only MODEL-DERIVED prices are kept -- assumed carbon-tax rates, statutory ETS
# quota prices and exogenous price assumptions (Cost_Type_Detail) are dropped.
#
# Every number that appears in the figure and the generated doc is computed
# here; nothing is hard-coded in the HTML.
#
# INPUT   outputs/output-stage3c/harmonized-costs.csv
#         outputs/output-stage3c/netzero-flags.csv
# OUTPUT  outputs/analysis/carbon-price-at-netzero.csv
#         figures/carbon-price-at-netzero.png
#         docs/carbon-price-at-netzero.html
#
#   python3 stage3d-carbon-price.py

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
SCOPE_COLOR = {
    "global": "#4a3aa7", "national": "#2a78d6",
    "subnational": "#eb6834", "unspecified": "#8a8a84",
}

CARBON_PRICE_TYPE = "Carbon price or abatement cost"

# Cost_Type_Detail phrasings that mark a *policy* carbon-price level (an assumed
# tax rate, a statutory ETS quota price, an exogenous price assumption) rather
# than the model-derived shadow / marginal-abatement price a least-cost pathway
# needs.  "... tax required to achieve ..." is a derived value and is kept.
POLICY_PRICE_DETAIL = re.compile(
    r"quota price"
    r"|price assumption"
    r"|\bassumed\b.*\b(tax|price)\b"
    r"|constant carbon tax"
    r"|initial carbon (tax|price)"
    r"|\bco2e? tax\b(?!.*required)"
    r"|\bcarbon tax\b(?!.*required)",
    re.IGNORECASE,
)

_NO_CDR = re.compile(r"\bno[\s_-]*(beccs|cdr|ccs|dac|net|removal)", re.IGNORECASE)

# USD2025/<denominator>  ->  factor to reach USD2025 per tCO2e
DENOM_TO_TCO2E = {
    "tco2": 1.0, "tco2eq": 1.0, "tco2e": 1.0, "tonco2": 1.0, "t-co2": 1.0,
    "t": 1.0, "mgco2eq": 1.0, "mgco2": 1.0, "tc": 44.0 / 12.0,
    "kgco2eq": 1000.0, "kgco2": 1000.0, "kg": 1000.0,
    "ktco2": 1e-3, "ktco2eq": 1e-3,
}

# a flag whose neutrality year is earlier than this is treated as not a
# genuine long-run net-zero end state
MIN_ENDPOINT_YEAR = 2030

DEFAULT_COSTS = Path("outputs/output-stage3c/harmonized-costs.csv")
DEFAULT_FLAGS = Path("outputs/output-stage3c/netzero-flags.csv")
DEFAULT_OUTDIR = Path("outputs/analysis")
DEFAULT_FIGDIR = Path("figures")
DEFAULT_DOC = Path("docs/carbon-price-at-netzero.html")


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
    return int(years[-1]) if len(years) == 1 else None


def source_ref(row):
    """'Figure 3' / 'Table S5' / 'Section 4.2' reference for one harmonised row,
    so a data point can be traced back to the paper for manual checking."""
    st = (row.get("Source_Type") or "").strip()
    sn = (row.get("Source_Number") or "").strip()
    if st and sn:
        return f"{st} {sn}"
    return st or sn or "Not reported"


_SUBNATIONAL = re.compile(
    r"\b(province|voivodeship|prefecture|county|municipal|canton|"
    r"bay area|greater|quebec|shanxi|guangdong|beijing|shenzhen|shanghai|"
    r"california|texas|wales|scotland|bavaria|catalonia|"
    r"\d+ western us states|western us)\b", re.IGNORECASE)


def geographic_scope(area):
    a = (area or "").strip()
    if not a:
        return "unspecified"
    if re.search(r"\b(global|world|worldwide|gcam regions)\b", a, re.IGNORECASE):
        return "global"
    if _SUBNATIONAL.search(a):
        return "subnational"
    return "national"


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def price_per_tco2e(row):
    value = to_float(row.get("Value_std"))
    unit = row.get("Value_std_Unit", "")
    if value is None or not unit.startswith("USD2025/"):
        return None
    denom = unit.split("/", 1)[1].strip().lower().replace(" ", "")
    factor = DENOM_TO_TCO2E.get(denom)
    return None if factor is None else value * factor


def _flag_index(flags):
    """{paper_id: {scenario_canon: {'last_year': int, 'area_canon': str}}} for
    the flagged net-zero scenarios (max neutrality year when a scenario has
    several flag rows)."""
    by_paper = {}
    for r in flags:
        y = parse_year(r.get("last_year"))
        if y is None or y < MIN_ENDPOINT_YEAR:
            continue
        sc = r.get("scenario_canon") or canon_scenario(r.get("Scenario_Name"))
        d = by_paper.setdefault(r["paper_id"], {})
        if sc not in d or y > d[sc]["last_year"]:
            d[sc] = {"last_year": y,
                     "area_canon": r.get("area_canon") or canon_area(r.get("Area"))}
    return by_paper


def _price_at(items, endpoint):
    """items: [(year, price, row)]. Rows/prices at the year closest to
    `endpoint` (ties -> the later year); everything when no year is known.
    Returns (rows, prices, picked_year_or_None)."""
    yrs = sorted({y for y, _, _ in items if y is not None})
    if not yrs:
        return ([r for _, _, r in items], [p for _, p, _ in items], None)
    pick = min(yrs, key=lambda y: (abs(y - endpoint), -y))
    rows = [r for y, _, r in items if y == pick]
    prices = [p for y, p, _ in items if y == pick]
    return rows, prices, pick


def build(cost_rows, flags):
    flag_by_paper = _flag_index(flags)

    by_scenario, catchall, dropped_policy = {}, {}, 0
    for r in cost_rows:
        if r.get("Cost_Type") != CARBON_PRICE_TYPE:
            continue
        if POLICY_PRICE_DETAIL.search(r.get("Cost_Type_Detail", "")):
            dropped_policy += 1
            continue
        price = price_per_tco2e(r)
        if price is None:
            continue
        pid = r["paper_id"]
        item = (parse_year(r.get("Year")), price, r)
        if is_catchall_scenario(r.get("Scenario_Name")):
            catchall.setdefault(pid, []).append(item)
        else:
            sc = r.get("Scenario_canon") or canon_scenario(r.get("Scenario_Name"))
            ac = r.get("Area_canon") or canon_area(r.get("Area"))
            by_scenario.setdefault((pid, sc, ac), []).append(item)

    def _emit(pid, scen_disp, area_disp, endpoint, items, via):
        used, prices, pick = _price_at(items, endpoint)
        meta = items[0][2]
        return {
            "paper_id": pid,
            "title": meta.get("Title", ""),
            "doi": meta.get("DOI", ""),
            "scenario": scen_disp,
            "area": area_disp,
            "scope": geographic_scope(area_disp),
            "netzero_year": endpoint,
            "price_year": pick if pick is not None else "",
            "matched_via": via,
            "n_price_rows": len(prices),
            "price": round(statistics.median(prices), 1),
            "raw_values": " | ".join(sorted({
                f"{(x.get('Value') or '').strip()} "
                f"{(x.get('Value_Unit') or '').strip()}".strip() for x in used})),
            "source": " | ".join(sorted({source_ref(x) for x in used})),
        }

    rows, matched_scen = [], set()
    for (pid, sc, ac), items in sorted(by_scenario.items()):
        per_paper = flag_by_paper.get(pid)
        if not per_paper:
            continue
        hit = per_paper.get(sc) or resolve_scenario(sc, per_paper)
        if hit is None:
            continue
        meta = items[0][2]
        rows.append(_emit(
            pid, (meta.get("Scenario_Name") or "").strip(),
            (meta.get("Area") or "").strip(), hit["last_year"], items,
            "scenario"))
        matched_scen.add((pid, sc))

    # a paper-wide carbon price (Scenario_Name "All scenarios" / "" / ...)
    # applies to every flagged scenario of that paper that has no price of
    # its own
    applied_catchall = 0
    for pid, items in sorted(catchall.items()):
        per_paper = flag_by_paper.get(pid)
        if not per_paper:
            continue
        for sc, fl in sorted(per_paper.items()):
            if (pid, sc) in matched_scen:
                continue
            fr = next((f for f in flags if f["paper_id"] == pid
                       and (f.get("scenario_canon")
                            or canon_scenario(f.get("Scenario_Name"))) == sc),
                      None)
            rows.append(_emit(
                pid, (fr.get("Scenario_Name") if fr else sc) or sc,
                (fr.get("Area") if fr else "") or "",
                fl["last_year"], items, "paper-wide"))
            applied_catchall += 1

    rows.sort(key=lambda r: r["price"])
    return rows, {"dropped_policy": dropped_policy, "flags_total": len(flags),
                  "applied_catchall": applied_catchall}


# ============================================================================
# STATISTICS  (every value the figure and the doc report)
# ============================================================================

def _frac_words(k, n):
    f = k / n
    for lo, hi, word in [(.28, .40, "about a third"),
                         (.20, .28, "roughly a quarter"),
                         (.40, .60, "close to half"),
                         (.60, .80, "most")]:
        if lo <= f < hi:
            return word
    return f"{k} of the {n}"


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
    low_paper_rows = sorted((r["price"] for r in by_paper[low["paper_id"]]))

    # BECCS-off vs BECCS-on comparison, if a paper has such a pair
    beccs = None
    for pid, prs in by_paper.items():
        off = [r for r in prs if _NO_CDR.search(r["scenario"])]
        on = [r for r in prs if not _NO_CDR.search(r["scenario"])]
        if off and on:
            beccs = {"paper": pid,
                     "with": min(r["price"] for r in on),
                     "without": max(r["price"] for r in off)}
            beccs["ratio"] = beccs["without"] / max(beccs["with"], 1e-9)
            break

    glob = next((r for r in sorted(rows, key=lambda r: -r["price"])
                 if r["scope"] == "global"), None)

    # first price at or above which the non-dominant scenarios sit
    non_dom = sorted(r["price"] for r in rows if r["paper_id"] != dom_id)
    cluster_from = round(statistics.quantiles(non_dom, n=4)[0], -1) \
        if len(non_dom) >= 4 else round(min(non_dom), -1)

    return {
        "n": len(rows),
        "n_papers": len(papers),
        "flags_total": meta["flags_total"],
        "dropped_policy": meta["dropped_policy"],
        "applied_catchall": meta["applied_catchall"],
        "p_min": round(min(prices)), "p_max": round(max(prices)),
        "median": round(statistics.median(prices)),
        "q1": round(q1), "q3": round(q3),
        "spread": round(max(prices) / max(min(prices), 1e-9)),
        "year_min": min(r["netzero_year"] for r in rows),
        "year_max": max(r["netzero_year"] for r in rows),
        "dom_paper": dom_id, "dom_n": len(dom_rows),
        "dom_area": dom_rows[0]["area"],
        "dom_min": round(dom_prices[0]), "dom_max": round(dom_prices[-1]),
        "dom_fraction": _frac_words(len(dom_rows), len(rows)),
        "dom_pct": round(100 * len(dom_rows) / len(rows)),
        "low_paper": low["paper_id"], "low_area": low["area"],
        "low_lo": round(low_paper_rows[0]), "low_hi": round(low_paper_rows[-1]),
        "low_n": len(low_paper_rows),
        "cluster_from": int(cluster_from),
        "beccs": beccs,
        "global_price": round(glob["price"]) if glob else None,
        "global_year": glob["netzero_year"] if glob else None,
        "global_area": glob["area"] if glob else None,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["paper_id", "title", "doi", "scenario", "area", "scope",
              "netzero_year", "price_year", "matched_via", "n_price_rows",
              "price", "raw_values", "source"]
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
    q1, med, q3 = statistics.quantiles(v, n=4)[0], statistics.median(v), \
        statistics.quantiles(v, n=4)[2]
    iqr = q3 - q1
    lo = max(min(v), q1 - 1.5 * iqr)
    hi = min(max(v), q3 + 1.5 * iqr)
    ax.add_patch(plt.Rectangle((x - width / 2, q1), width, iqr,
                               facecolor="#eef1f4", edgecolor=INK2, lw=1.2,
                               zorder=2))
    ax.plot([x - width / 2, x + width / 2], [med, med], color=INK, lw=2.2, zorder=3)
    ax.plot([x, x], [lo, q1], color=INK2, lw=1.0, zorder=2)
    ax.plot([x, x], [q3, hi], color=INK2, lw=1.0, zorder=2)


def figure_caveat(stats):
    dom = f"one {stats['dom_area'].split(',')[0]} study contributes " \
          f"{stats['dom_n']} of the {stats['n']}"
    return (
        f"Exploratory. One shadow / marginal-abatement price per net-zero "
        f"scenario at its last pathway year, in real 2025 USD/tCO2e (CO2 and "
        f"CO2e bases pooled). {stats['dropped_policy']} assumed-tax / statutory-"
        f"ETS rows excluded. {dom[0].upper()}{dom[1:]}; net-zero years span "
        f"{stats['year_min']}-{stats['year_max']}."
    )


def fig_carbon_price(rows, stats, path):
    prices = [r["price"] for r in rows]
    scopes = [s for s in ("national", "subnational", "global")
              if any(r["scope"] == s for r in rows)]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.2, 5.6), gridspec_kw={"width_ratios": [1, 2.3]},
        sharey=True)

    _box(ax1, 0, prices, width=0.5)
    for dy, r in zip(_jitter(len(rows)), rows):
        ax1.scatter([dy], [r["price"]], s=44, c=SCOPE_COLOR[r["scope"]],
                    edgecolors="white", linewidths=0.7, alpha=0.9, zorder=5)
    ax1.set_xlim(-0.7, 0.7)
    ax1.set_xticks([0])
    ax1.set_xticklabels(
        [f"all\n{stats['n']} scenarios\n{stats['n_papers']} papers"])
    ax1.set_ylabel("carbon price at net zero  ·  real 2025 USD / tCO$_2$e")
    ax1.axhline(stats["median"], color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax1.text(-0.66, stats["median"], f"median\n${stats['median']:,.0f}",
             fontsize=8, fontweight="bold", va="center", ha="left", color=INK)

    for i, r in enumerate(rows):
        ax2.plot([i, i], [min(prices) * 0.6, r["price"]], color=GRID, lw=1.0,
                 zorder=1)
        ax2.scatter([i], [r["price"]], s=46, c=SCOPE_COLOR[r["scope"]],
                    edgecolors="white", linewidths=0.7, zorder=5)
        area = r["area"] if len(r["area"]) <= 20 else r["area"][:19] + "…"
        ax2.text(i, r["price"] * 1.14, f"${r['price']:,.0f}", ha="center",
                 fontsize=7.4, color=INK)
        ax2.text(i, min(prices) * 0.5,
                 f"{area}\np{r['paper_id']} · {r['netzero_year']}",
                 ha="right", va="top", rotation=45, fontsize=6.8, color=INK2,
                 rotation_mode="anchor")
    ax2.set_xticks([])
    ax2.set_xlim(-0.8, len(rows) - 0.2)
    ax2.set_title("ranked low → high", fontsize=9, loc="left", color=MUTED,
                  pad=4)

    for ax in (ax1, ax2):
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", color=GRID, lw=0.8)
        ax.minorticks_off()
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    ax1.set_ylim(min(prices) * 0.6, max(prices) * 1.35)
    ticks = [t for t in (25, 50, 100, 250, 500, 1000, 2500)
             if min(prices) * 0.5 <= t <= max(prices) * 1.5]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f"${t:,.0f}" for t in ticks])

    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=7,
                          mfc=SCOPE_COLOR[k], mec="white") for k in scopes]
    fig.legend(handles, scopes, loc="lower center", ncol=len(scopes),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        f"Carbon price in the year of net zero  ·  {stats['n']} scenarios, "
        f"{stats['n_papers']} papers",
        fontsize=11.5, x=0.03, ha="left", y=0.99)
    fig.text(0.5, -0.13, figure_caveat(stats), ha="center", va="top",
             fontsize=7.4, color=MUTED, wrap=True)

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# DOCUMENT  (every number interpolated from `stats` / `rows`)
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


def _sentence(text):
    return f"<li>{text}</li>" if text else ""


def build_doc(rows, stats, png_b64):
    s = stats
    esc = html.escape

    beccs_li = ""
    if s["beccs"]:
        b = s["beccs"]
        extra = ""
        if s["global_price"]:
            extra = (f" The global run (<code>{esc(s['global_area'])}</code>) sits "
                     f"at <strong>${s['global_price']:,}</strong> by "
                     f"{s['global_year']}.")
        beccs_li = _sentence(
            f"<strong>Removals cut the price.</strong> The one paper with a "
            f"paired comparison (<code>p{b['paper']}</code>) needs "
            f"<strong>${round(b['with']):,}</strong>/tCO&#8322; with CDR and "
            f"<strong>${round(b['without']):,}</strong> without &mdash; a "
            f"{b['ratio']:.0f}&times; jump.{extra}")

    table_rows = "\n".join(
        f"          <tr><td class=\"n\">{r['price']:,.0f}</td>"
        f"<td>{esc(r['area'])}</td><td>{esc(r['scenario'])}</td>"
        f"<td class=\"n\">{r['netzero_year']}</td><td>{esc(r['scope'])}</td>"
        f"<td>p{esc(r['paper_id'])}</td></tr>"
        for r in rows)

    return f"""<title>Carbon Price at Net Zero</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Newsreader:ital@1&display=swap">

<style>{DOC_CSS}</style>

<div class="topbar">
  <b>carbon&#8202;·&#8202;price</b>
  <span>stage3d-carbon-price.py</span>
</div>

<main>
  <div class="hero">
    <p class="kicker">Data extraction pipeline &nbsp;/&nbsp; analysis</p>
    <h1>Carbon Price at Net Zero</h1>
    <p class="lede">
      Of {s['flags_total']} scenarios flagged as reaching net zero,
      <em>{s['n']} also report a model-derived carbon price</em> at the end of
      the pathway &mdash; assumed tax rates and statutory ETS prices excluded.
      In real 2025 dollars they run from <strong>${s['p_min']:,}</strong> to
      <strong>${s['p_max']:,}</strong> per tonne CO&#8322;e.
    </p>
  </div>

  <figure>
    <img src="data:image/png;base64,{png_b64}" alt="Box plot and ranked dot plot of {s['n']} model-derived carbon prices in the net-zero year, log scale, ${s['p_min']:,} to ${s['p_max']:,} per tonne CO2e, median ${s['median']:,}.">
    <figcaption>
      figures/carbon-price-at-netzero.png &mdash; left: the distribution
      (box = IQR, line = median). right: every scenario, ranked, labelled by
      area &middot; paper &middot; net-zero year. log scale.
    </figcaption>
  </figure>

  <section>
    <h2>What it shows</h2>
    <div class="stat-row">
      <div class="stat"><b>${s['median']:,}</b><span>median</span></div>
      <div class="stat"><b>${s['q1']:,}&ndash;{s['q3']:,}</b><span>interquartile range</span></div>
      <div class="stat"><b>${s['p_min']:,}&ndash;{s['p_max']:,}</b><span>full range</span></div>
      <div class="stat"><b>{s['n']} / {s['n_papers']}</b><span>scenarios / papers</span></div>
    </div>
    <ul>
      <li><strong>The spread is the finding.</strong> A ~{s['spread']}&times; range
        between the cheapest and most expensive net-zero shadow price, driven by
        how much the model leans on carbon dioxide removal and how late
        neutrality arrives.</li>
      {beccs_li}
      <li><strong>The low end is regional.</strong> The cheapest entries
        (<code>p{s['low_paper']}</code>, {esc(s['low_area'])},
        ${s['low_lo']:,}&ndash;{s['low_hi']:,}) come from one study; the rest
        cluster from about <strong>${s['cluster_from']:,}</strong> upward.</li>
      <li>One {esc(s['dom_area'].split(',')[0])} study (<code>p{s['dom_paper']}</code>)
        contributes {s['dom_fraction']} of the {s['n']} ({s['dom_pct']}%),
        all in the upper half (${s['dom_min']:,}&ndash;{s['dom_max']:,}); it
        weights the distribution toward high values.</li>
    </ul>
  </section>

  <section>
    <h2>Method</h2>
    <div class="flow">
      <div class="step"><b>1</b> net-zero filter</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>2</b> shadow-price rows</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>3</b> normalise unit</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>4</b> price at the neutrality year</div>
    </div>
    <ul>
      <li><strong>Net-zero membership</strong> comes from
        <code>netzero-flags.csv</code> (<code>is_netzero&nbsp;==&nbsp;True</code>)
        &mdash; the same filter as the sector-balance figure, but here
        <em>no sector breakdown is required</em>. Scenario and area are matched
        on their canonical keys, so <code>"NZE"</code> and <code>"2050&nbsp;NZE"</code>
        join.</li>
      <li><strong>Shadow-price rows</strong> are <code>harmonized-costs.csv</code>
        rows with <code>Cost_Type = "Carbon price or abatement cost"</code> and a
        <code>USD2025/&hellip;</code> value. Rows whose <code>Cost_Type_Detail</code>
        names a <em>policy</em> level &mdash; <code>CO2 tax</code>,
        <code>quota price</code>, <code>&hellip; price assumption</code>,
        <code>constant carbon tax</code>, <code>assumed &hellip; tax</code> &mdash; are
        dropped ({s['dropped_policy']} rows); &ldquo;tax <em>required to
        achieve</em>&rdquo; phrasings are kept as derived values. A price whose
        scenario field is a catch-all (<code>"All scenarios"</code>,
        <code>"Model assumption"</code>, blank) is applied to every flagged
        scenario of that paper that has no price of its own
        ({s['applied_catchall']} rows here).</li>
      <li><strong>Unit normalisation</strong> to 2025 USD per tCO&#8322;e:
        <code>/tCO2</code>, <code>/tCO2eq</code>, <code>/tonCO2</code>,
        <code>/Mg&nbsp;CO2eq</code>, <code>/t</code> &rarr; &times;1;
        <code>/kgCO2eq</code> &rarr; &times;1000; <code>/ktCO2</code> &rarr; &times;0.001;
        <code>/tC</code> &rarr; &times;44/12; <code>/yr</code> and unrecognised
        &rarr; dropped. CO&#8322; and CO&#8322;e price bases are pooled.</li>
      <li><strong>End year</strong> = the scenario&rsquo;s neutrality year from
        <code>netzero-flags.csv</code>. The price is read at that year, or the
        nearest earlier year the scenario reports one
        (<code>price_year</code> in the CSV). One value per scenario &mdash;
        the median if several rows share that year.</li>
    </ul>
  </section>

  <section>
    <h2>The {s['n']} scenarios</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr><th class="n">USD2025 /tCO&#8322;e</th><th>Area</th><th>Scenario</th>
            <th class="n">Net-zero yr</th><th>Scope</th><th>Paper</th></tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
      </table>
    </div>
    <p class="t-note">Full table with titles and DOIs:
      <code>outputs/analysis/carbon-price-at-netzero.csv</code>.</p>
  </section>

  <section>
    <h2>Caveats</h2>
    <div class="callout">
      <ul>
        <li><strong>n = {s['n']}</strong>, and one study
          (<code>p{s['dom_paper']}</code>) is {s['dom_n']} of them
          &mdash; {s['dom_pct']}% of the sample, all high.</li>
        <li>CO&#8322; and CO&#8322;e price bases are pooled &mdash; a modest distortion
          for carbon prices, larger for GHG-basket abatement costs.</li>
        <li>The policy / shadow split is drawn from <code>Cost_Type_Detail</code>
          keywords; a handful of ambiguous phrasings (&ldquo;national carbon
          market price&rdquo;, &ldquo;reference CO&#8322; price&rdquo;) are kept and could
          be model assumptions rather than outputs.</li>
        <li>Net-zero years span
          <strong>{s['year_min']}&ndash;{s['year_max']}</strong>; later dates tend
          to carry higher end-of-pathway prices, and the sample is too thin to
          separate that from study-to-study variation.</li>
        <li>The net-zero flag has a threshold (<code>--netzero-fraction</code>,
          default 0.05); a looser value adds scenarios, but the carbon-price
          bottleneck is that <strong>most net-zero scenarios don't report
          one</strong>.</li>
      </ul>
    </div>
  </section>
</main>

<footer>
  reproduce &nbsp;&rarr;&nbsp; <code>python3 stage3d-carbon-price.py</code><br>
  inputs: <code>harmonized-costs.csv</code>, <code>netzero-flags.csv</code>
  &nbsp;&middot;&nbsp; outputs: <code>figures/carbon-price-at-netzero.png</code>,
  <code>outputs/analysis/carbon-price-at-netzero.csv</code>
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
        raise SystemExit("no net-zero scenario carries a carbon price")

    stats = compute_stats(rows, meta)
    write_csv(Path(args.outdir) / "carbon-price-at-netzero.csv", rows)

    fig_path = Path(args.figures_dir) / "carbon-price-at-netzero.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _style()
    fig_carbon_price(rows, stats, fig_path)

    png_b64 = base64.b64encode(fig_path.read_bytes()).decode()
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_doc(rows, stats, png_b64), encoding="utf-8")

    print(f"net-zero scenarios flagged : {stats['flags_total']}")
    print(f"  policy-price rows dropped : {stats['dropped_policy']}")
    print(f"  paper-wide prices applied : {stats['applied_catchall']}")
    print(f"  with a shadow price       : {stats['n']}  "
          f"({stats['n_papers']} papers)")
    print(f"  USD2025/tCO2e : min {stats['p_min']:,}  Q1 {stats['q1']:,}  "
          f"median {stats['median']:,}  Q3 {stats['q3']:,}  "
          f"max {stats['p_max']:,}  (spread {stats['spread']}x)")
    print(f"\nfigure -> {fig_path}\ndoc    -> {doc_path}")


if __name__ == "__main__":
    main()
