# scripts/09-stage3c-capex-by-technology.py  (stage 3c -- exploratory analysis)
#
# What overnight capital cost (CAPEX) do the reviewed net-zero studies assume
# for each electricity-supply technology, and how wide is the disagreement?
#
# Unlike the carbon-price / electricity-price figures this one is NOT filtered
# to net-zero scenarios -- a CAPEX number is a techno-economic *input*, usually
# stated once for the whole study, so every USD2025/MW value in
# harmonized-costs.csv is used.
#
# Each cost row with Value_std_Unit == "USD2025/MW" is:
#   - de-duplicated on (paper, detail, year, value)
#   - classified to a technology from Cost_Type_Detail / Sector_Other (TECHMAP)
#   - tagged near-term (year <= 2030) / long-term (year >= 2035) / unspecified
# Fixed-O&M rows (USD2025/MW.yr) and non-generation equipment (boilers, heat
# pumps, lighting, motors) are excluded.
#
# Technologies with < MIN_ROWS values or < MIN_PAPERS papers are dropped from
# the plot (listed in the doc). The figure is a horizontal box + strip plot,
# one row per technology, log x-axis, points coloured by time horizon.
#
# Every number in the figure and the doc is computed here.
#
# INPUT   outputs/output-stage3c/harmonized-costs.csv
# OUTPUT  outputs/analysis/capex-by-technology.csv
#         figures/capex-by-technology.png
#         docs/capex-by-technology.html
#
#   python3 scripts/09-stage3c-capex-by-technology.py   (run from the repo root)

from pathlib import Path
import argparse
import base64
import csv
import html
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID = "#1a1a1a", "#55534e", "#8a8a84", "#dedcd5"

# Points are coloured by world region and shaped by cost year (time horizon).
HORIZON_ORDER = ["near-term", "long-term", "unspecified"]
HORIZON_MARKER = {"near-term": "o", "long-term": "^", "unspecified": "s"}

# Fixed region order + a colourblind-safe hue each (Okabe-Ito based); only
# regions present get a legend entry. Shared verbatim with
# 08-stage3c-electricity-price.py.
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

_REGION_RULES = [
    ("Global / multi-region",
     r"\b(global|world(wide)?|multi-?region|gcam regions|all regions|"
     r"model regions|whole model|aggregate)\b"),
    ("China",
     r"\bchina\b|\bprc\b|chinese|nanning|shanxi|guangdong|guangxi|"
     r"beijing|shanghai|shenzhen|sichuan|inner mongolia|yangtze|"
     r"^(ea|na|sa|wa|nc|sc|ec|wc)$"),
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

MIN_ROWS = 5       # a technology needs this many values ...
MIN_PAPERS = 2     # ... from at least this many papers to get its own row

# Cost_Type_Detail / Sector_Other  ->  technology.  First match wins, so the
# specific patterns (CCS, offshore) sit above the generic ones.
TECHMAP = [
    ("Battery storage",      r"\b(battery|li-?ion|lithium|bess)\b"),
    ("Electrolyser",         r"electroly[sz]"),
    ("Gas + CCS",            r"(gas|ccgt|combined.cycle|natural.?gas).*"
                             r"(ccs|carbon captur|with cc)|ccs.*gas"),
    ("Coal + CCS",           r"coal.*(ccs|carbon captur)|ccs.*coal"),
    ("Concentrated solar",   r"\bcsp\b|concentrat\w+ solar|solar thermal"),
    ("Solar PV",             r"solar pv|photovoltaic|\bpv\b|solar power|"
                             r"distributed solar|utility.scale solar|rooftop"),
    ("Onshore wind",         r"onshore wind|wind onshore"),
    ("Offshore wind",        r"offshore wind|wind offshore"),
    ("Wind (unspecified)",   r"\bwind\b"),
    ("Nuclear",              r"\bnuclear\b|light.water react|generation ii|"
                             r"generation iii|\bsmr\b"),
    ("Hydropower",           r"\bhydro(power|electric|-electric)?\b|"
                             r"pumped.storage hydro"),
    ("Geothermal",           r"\bgeothermal\b"),
    ("Bioenergy",            r"\b(biomass|bioenerg\w+|biopower|biogas|"
                             r"bio-?ccs|beccs)\b"),
    ("Hydrogen turbine/FC",  r"hydrogen (generation|electric|turbine|"
                             r"fuel.?cell)|h2 (turbine|generation)"),
    ("Gas (unabated)",       r"\b(gas combined.cycle|gas steam|natural.?gas|"
                             r"ccgt|ocgt|gas turbine|gas.fired|gas power|"
                             r"gas generation|gas plant)\b"),
    ("Coal (unabated)",      r"\bcoal\b"),
    ("Liquid-fuel power",    r"liquid.?fuel|refined liquids|oil.fired|diesel gen"),
    ("Wave / tidal",         r"\bwave\b|\btidal\b"),
    ("Municipal-waste power", r"municipal.?waste|waste.to.energy|waste power"),
    ("CHP",                  r"\bchp\b|combined heat and power|cogener"),
]

DROP_DETAIL = re.compile(
    r"o&m|opex|operation.and.mainten|fixed operation"
    r"|energy supply cost|lighting equipment|motor equipment"
    r"|\bboiler\b|heat pump|power-to-gas",
    re.IGNORECASE,
)

DEFAULT_COSTS = Path("outputs/output-stage3c/harmonized-costs.csv")
DEFAULT_OUTDIR = Path("outputs/analysis")
DEFAULT_FIGDIR = Path("figures")
DEFAULT_DOC = Path("docs/capex-by-technology.html")


# ============================================================================
# LOADING / BUILDING
# ============================================================================

def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_year(raw):
    years = re.findall(r"(?:19|20)\d{2}", (raw or "").strip())
    return int(years[-1]) if years else None


def source_ref(row):
    st = (row.get("Source_Type") or "").strip()
    sn = (row.get("Source_Number") or "").strip()
    return f"{st} {sn}".strip() or "Not reported"


def classify(row):
    s = " ".join((row.get("Cost_Type_Detail") or "",
                  row.get("Sector_Other") or "",
                  row.get("Scenario_Name") or "")).lower()
    for name, pat in TECHMAP:
        if re.search(pat, s):
            return name
    return None


def horizon(year):
    if year is None:
        return "unspecified"
    if year <= 2030:
        return "near-term"
    if year >= 2035:
        return "long-term"
    return "unspecified"


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def build(cost_rows):
    seen, points = set(), []
    n_mw_rows = dropped_detail = unclassified = 0

    for r in cost_rows:
        if (r.get("Value_std_Unit") or "").strip() != "USD2025/MW":
            continue
        n_mw_rows += 1
        detail = r.get("Cost_Type_Detail") or ""
        if DROP_DETAIL.search(detail + " " + (r.get("Sector_Other") or "")):
            dropped_detail += 1
            continue
        value = to_float(r.get("Value_std"))
        if value is None or value <= 0:
            continue
        key = (r["paper_id"], detail.strip().lower(),
               (r.get("Year") or "").strip(), r.get("Value_std"))
        if key in seen:
            continue
        seen.add(key)
        tech = classify(r)
        if tech is None:
            unclassified += 1
            continue
        year = parse_year(r.get("Year"))
        points.append({
            "paper_id": r["paper_id"],
            "title": r.get("Title", ""),
            "doi": r.get("DOI", ""),
            "technology": tech,
            "capex_musd_per_mw": round(value / 1e6, 4),
            "capex_usd_per_kw": round(value / 1e3, 1),
            "year": year if year is not None else "",
            "horizon": horizon(year),
            "scenario": (r.get("Scenario_Name") or "").strip(),
            "area": (r.get("Area") or "").strip(),
            "region": world_region(r.get("Area")),
            "detail": detail.strip(),
            "raw_value": f"{(r.get('Value') or '').strip()} "
                         f"{(r.get('Value_Unit') or '').strip()}".strip(),
            "source": source_ref(r),
        })

    by_tech = {}
    for p in points:
        by_tech.setdefault(p["technology"], []).append(p)

    kept, dropped_small = {}, []
    for tech, ps in by_tech.items():
        n_papers = len({p["paper_id"] for p in ps})
        if len(ps) >= MIN_ROWS and n_papers >= MIN_PAPERS:
            kept[tech] = ps
        else:
            dropped_small.append((tech, len(ps), n_papers))

    order = sorted(kept, key=lambda t: statistics.median(
        p["capex_musd_per_mw"] for p in kept[t]))

    meta = {
        "n_mw_rows": n_mw_rows,
        "dropped_detail": dropped_detail,
        "unclassified": unclassified,
        "n_points_kept": sum(len(v) for v in kept.values()),
        "dropped_small": sorted(dropped_small, key=lambda x: -x[1]),
    }
    return points, kept, order, meta


# ============================================================================
# STATISTICS
# ============================================================================

def compute_stats(points, kept, order, meta):
    per_tech = {}
    for tech in order:
        vals = sorted(p["capex_musd_per_mw"] for p in kept[tech])
        q1, med, q3 = (statistics.quantiles(vals, n=4)[0]
                       if len(vals) >= 2 else vals[0],
                       statistics.median(vals),
                       statistics.quantiles(vals, n=4)[2]
                       if len(vals) >= 2 else vals[0])
        per_tech[tech] = {
            "n": len(vals),
            "n_papers": len({p["paper_id"] for p in kept[tech]}),
            "min": vals[0], "max": vals[-1],
            "median": med, "q1": q1, "q3": q3,
            "spread": round(vals[-1] / vals[0], 1),
        }

    all_med = {t: per_tech[t]["median"] for t in order}
    cheapest = min(all_med, key=all_med.get)
    dearest = max(all_med, key=all_med.get)

    # near vs long term, same technology, median shift
    learning = []
    for tech in order:
        near = [p["capex_musd_per_mw"] for p in kept[tech]
                if p["horizon"] == "near-term"]
        long = [p["capex_musd_per_mw"] for p in kept[tech]
                if p["horizon"] == "long-term"]
        if len(near) >= 2 and len(long) >= 2:
            learning.append((tech, statistics.median(near),
                             statistics.median(long)))
    learning_drop = None
    if learning:
        drops = [(t, n, l, (n - l) / n) for t, n, l in learning if n > 0]
        drops.sort(key=lambda x: -x[3])
        learning_drop = drops[0]

    widest = max(order, key=lambda t: per_tech[t]["spread"])

    kept_pts = [p for p in points if p["technology"] in kept]
    by_region = {}
    for p in kept_pts:
        by_region.setdefault(p["region"], []).append(p["capex_musd_per_mw"])
    regions = [g for g in REGION_ORDER if g in by_region]
    region_summary = {g: {"n": len(by_region[g]),
                          "median": round(statistics.median(by_region[g]), 2)}
                      for g in regions}

    return {
        "n_points": meta["n_points_kept"],
        "n_tech": len(order),
        "n_papers": len({p["paper_id"] for p in points}),
        "n_mw_rows": meta["n_mw_rows"],
        "dropped_detail": meta["dropped_detail"],
        "unclassified": meta["unclassified"],
        "dropped_small": meta["dropped_small"],
        "per_tech": per_tech,
        "order": order,
        "cheapest": cheapest, "cheapest_med": all_med[cheapest],
        "dearest": dearest, "dearest_med": all_med[dearest],
        "widest": widest, "widest_spread": per_tech[widest]["spread"],
        "learning_drop": learning_drop,
        "regions": regions,
        "region_summary": region_summary,
        "horizons_present": [h for h in HORIZON_ORDER
                             if any(p["horizon"] == h for p in kept_pts)],
    }


def write_csv(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["paper_id", "title", "doi", "technology", "capex_musd_per_mw",
              "capex_usd_per_kw", "year", "horizon", "region", "scenario",
              "area", "detail", "raw_value", "source"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(points, key=lambda p: (p["technology"],
                                                  p["capex_musd_per_mw"])))


# ============================================================================
# FIGURE
# ============================================================================

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": GRID, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 160,
    })


def _jitter(k, spread=0.13):
    if k <= 1:
        return [0.0]
    step = (2 * spread) / (k - 1)
    return [(-spread + i * step) for i in range(k)]


def fig_capex(kept, order, stats, path):
    n = len(order)
    fig, ax = plt.subplots(figsize=(9.6, 0.58 * n + 2.4))

    all_vals = sorted(p["capex_musd_per_mw"] for ps in kept.values() for p in ps)
    x_lo = all_vals[0] * 0.75
    # clip the display to the 96th percentile so a couple of BECCS-scale
    # outliers don't compress every other row (they still plot, clipped)
    x_hi = all_vals[min(len(all_vals) - 1, int(len(all_vals) * 0.96))] * 1.5
    n_clipped = sum(1 for v in all_vals if v > x_hi)

    for y, tech in enumerate(order):
        st = stats["per_tech"][tech]
        q1, med, q3 = st["q1"], st["median"], st["q3"]

        # faint full-range guide + IQR bar + median tick
        ax.plot([st["min"], st["max"]], [y, y], color=GRID, lw=1.0, zorder=1)
        ax.plot([q1, q3], [y, y], color=INK2, lw=3.0,
                solid_capstyle="butt", zorder=3, alpha=0.35)
        ax.plot([med, med], [y - 0.26, y + 0.26], color=INK, lw=2.4, zorder=5)

        ps = sorted(kept[tech], key=lambda p: p["capex_musd_per_mw"])
        for dy, p in zip(_jitter(len(ps)), ps):
            ax.scatter([p["capex_musd_per_mw"]], [y + dy], s=28,
                       marker=HORIZON_MARKER[p["horizon"]],
                       c=REGION_COLOR[p["region"]], edgecolors="white",
                       linewidths=0.5, alpha=0.92, zorder=6)

        ax.text(med, y + 0.32, f"${med:.2f}M", ha="center", va="bottom",
                fontsize=7.0, color=INK, zorder=7)

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{t}\n{stats['per_tech'][t]['n']} values · "
                        f"{stats['per_tech'][t]['n_papers']} papers"
                        for t in order], fontsize=8)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("overnight capital cost  ·  million real-2025 USD per MW  "
                  "(log scale)")
    xt = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32) if x_lo <= t <= x_hi]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"${t:g}M" for t in xt])
    ax.grid(axis="x", which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0)

    # one legend below the plot: colour = world region, shape = cost year
    reg = [plt.Line2D([0], [0], marker="o", ls="", ms=7,
                      mfc=REGION_COLOR[g], mec="white")
           for g in stats["regions"]]
    shp = [plt.Line2D([0], [0], marker=HORIZON_MARKER[h], ls="", ms=7,
                      mfc=MUTED, mec="white")
           for h in stats["horizons_present"]]
    shp.append(plt.Line2D([0], [0], color=INK, lw=2.4))
    reg_labels = list(stats["regions"])
    shp_labels = [f"{h} (cost yr)" for h in stats["horizons_present"]] + \
                 ["median"]
    fig.legend(reg + shp, reg_labels + shp_labels, loc="lower center",
               ncol=min(len(reg) + len(shp), 5), frameon=False, fontsize=7.8,
               bbox_to_anchor=(0.5, -0.06),
               title="colour = world region     ·     shape = cost year",
               title_fontsize=7.5)

    fig.suptitle(
        f"Assumed generation CAPEX by technology  ·  {stats['n_points']} "
        f"values, {stats['n_tech']} technologies, {stats['n_papers']} papers",
        fontsize=11.5, x=0.02, ha="left", y=0.995)
    fig.text(0.5, -0.12,
             "Exploratory. Each de-duplicated USD2025/MW value in "
             "harmonized-costs.csv, classified to a technology by its cost "
             f"label; {stats['dropped_detail']} O&M / equipment and "
             f"{stats['unclassified']} unclassifiable rows dropped. Grey bar "
             f"= IQR, tick = median. Axis clipped at ${x_hi:.0f}M "
             f"({n_clipped} values off-scale).",
             ha="center", va="top", fontsize=7, color=MUTED, wrap=True)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
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


def build_doc(stats, png_b64):
    s = stats
    esc = html.escape

    learn_li = ""
    if s["learning_drop"]:
        t, near, long, frac = s["learning_drop"]
        learn_li = (
            f"<li><strong>Learning is priced in.</strong> Where a study gives "
            f"both a near-term and a long-term number, the assumed cost falls "
            f"&mdash; {esc(t)} drops from <strong>${near:.2f}M</strong> to "
            f"<strong>${long:.2f}M</strong>/MW ({round(frac * 100)}% lower) "
            f"between the two horizons.</li>")

    tech_rows = "\n".join(
        f"          <tr><td>{esc(t)}</td>"
        f"<td class=\"n\">{d['median']:.2f}</td>"
        f"<td class=\"n\">{d['min']:.2f}&ndash;{d['max']:.2f}</td>"
        f"<td class=\"n\">{d['spread']}&times;</td>"
        f"<td class=\"n\">{d['n']}</td><td class=\"n\">{d['n_papers']}</td></tr>"
        for t, d in ((t, s["per_tech"][t]) for t in reversed(s["order"])))

    dropped = ", ".join(f"{esc(t)} ({nr}r/{npp}p)"
                        for t, nr, npp in s["dropped_small"]) or "none"

    region_li = " · ".join(
        f"{g} ${d['median']:.2f}M (n={d['n']})"
        for g, d in sorted(s["region_summary"].items(),
                           key=lambda kv: kv[1]["median"]))

    return f"""<title>CAPEX by Technology</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Newsreader:ital@1&display=swap">

<style>{DOC_CSS}</style>

<div class="topbar">
  <b>capex&#8202;·&#8202;by technology</b>
  <span>scripts/09-stage3c-capex-by-technology.py</span>
</div>

<main>
  <div class="hero">
    <p class="kicker">Data extraction pipeline &nbsp;/&nbsp; analysis</p>
    <h1>CAPEX by Technology</h1>
    <p class="lede">
      The reviewed net-zero studies state an overnight capital cost for the
      power technologies they deploy. Pooled across {s['n_papers']} papers,
      <em>{s['n_points']} values for {s['n_tech']} technologies</em> &mdash;
      and the studies disagree with each other by more than any single
      technology's cost falls over three decades.
    </p>
  </div>

  <figure>
    <img src="data:image/png;base64,{png_b64}" alt="Horizontal strip plot of assumed overnight CAPEX for {s['n_tech']} electricity technologies on a log axis, points coloured by world region and shaped by cost year.">
    <figcaption>
      figures/capex-by-technology.png &mdash; one row per technology, ordered
      by median. grey bar = IQR, black tick = median, points = individual
      study values. log x-axis. <strong>colour = world region</strong>;
      marker shape = cost year (circle near-term &le;&nbsp;2030, triangle
      long-term &ge;&nbsp;2035, square unspecified).
    </figcaption>
  </figure>

  <section>
    <h2>What it shows</h2>
    <div class="stat-row">
      <div class="stat"><b>{s['n_tech']}</b><span>technologies</span></div>
      <div class="stat"><b>{s['n_points']}</b><span>CAPEX values</span></div>
      <div class="stat"><b>{s['n_papers']}</b><span>papers</span></div>
      <div class="stat"><b>{s['widest_spread']}&times;</b><span>widest disagreement ({esc(s['widest'])})</span></div>
    </div>
    <ul>
      <li><strong>The ordering is conventional.</strong> Solar PV and onshore
        wind sit cheapest (median around
        ${s['per_tech'][s['cheapest']]['median']:.2f}M/MW), nuclear, geothermal
        and BECCS-type bioenergy most expensive (up to
        ${s['dearest_med']:.2f}M/MW for {esc(s['dearest'])}).</li>
      <li><strong>The within-technology spread is the story.</strong> Even for
        a mature technology the studies span a
        <strong>{s['widest_spread']}&times;</strong> range
        ({esc(s['widest'])}) &mdash; base year, region, financing assumptions
        and what the "plant" includes all differ and are not always
        recorded.</li>
      {learn_li}
      <li><strong>By world region</strong> (median across all technologies,
        so composition-weighted): {region_li}. Regions with only one or two
        contributing studies should not be read as a regional cost signal.</li>
    </ul>
  </section>

  <section>
    <h2>By technology</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr><th>Technology</th><th class="n">Median M$/MW</th>
            <th class="n">Range</th><th class="n">Spread</th>
            <th class="n">Values</th><th class="n">Papers</th></tr>
        </thead>
        <tbody>
{tech_rows}
        </tbody>
      </table>
    </div>
    <p class="t-note">Every value with its paper, year, region and original
      cost label: <code>outputs/analysis/capex-by-technology.csv</code>.</p>
  </section>

  <section>
    <h2>Method</h2>
    <div class="flow">
      <div class="step"><b>1</b> USD2025/MW rows</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>2</b> de-duplicate</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>3</b> classify technology</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>4</b> tag region + horizon</div>
    </div>
    <ul>
      <li><strong>Not net-zero filtered.</strong> A CAPEX number is a
        techno-economic <em>input</em>, usually stated once for the whole
        study, so every <code>USD2025/MW</code> row of
        <code>harmonized-costs.csv</code> is used
        ({s['n_mw_rows']} rows before cleaning).</li>
      <li><strong>Cleaning.</strong> Fixed-O&amp;M rows
        (<code>USD2025/MW.yr</code>) are a different unit and never enter here;
        non-generation equipment (boilers, heat pumps, lighting, motors) is
        dropped by keyword ({s['dropped_detail']} rows). Exact duplicates on
        <code>(paper, label, year, value)</code> are collapsed.</li>
      <li><strong>Technology</strong> is matched from
        <code>Cost_Type_Detail</code> / <code>Sector_Other</code> against a
        fixed pattern list &mdash; CCS and offshore variants are tested before
        the generic "gas" / "wind" patterns.
        {s['unclassified']} rows matched no pattern and are excluded.</li>
      <li><strong>Time horizon</strong> (marker shape) from the row's
        <code>Year</code>: near-term &le; 2030, long-term &ge; 2035, otherwise
        unspecified.</li>
      <li><strong>World region</strong> (marker colour) is keyword-matched
        from the <code>Area</code> string; a country or sub-national unit maps
        to its region, an unrecognised or blank area is
        <em>Not reported</em>.</li>
      <li><strong>Cut.</strong> A technology needs &ge; 5 values from &ge; 2
        papers to get its own row. Dropped for thinness: {dropped}.</li>
    </ul>
  </section>

  <section>
    <h2>Caveats</h2>
    <div class="callout">
      <ul>
        <li>These are <strong>assumptions, not outcomes</strong> &mdash; what
          each model was told a technology costs, not a result of the
          scenario.</li>
        <li>Currency base year is harmonised to 2025 USD, but the physical
          scope of "$/MW" is not: some values are bare equipment, others
          include grid connection, construction financing or storage.</li>
        <li>Region is pooled. A gas turbine in Saudi Arabia and one in
          Portugal are on the same row.</li>
        <li>Near-term / long-term is the row's stated cost year, which many
          studies leave blank &mdash; the unspecified points carry no learning
          information.</li>
        <li>Bioenergy mixes plain biomass combustion with BECCS full-chain
          costs; its high tail is partly that.</li>
      </ul>
    </div>
  </section>
</main>

<footer>
  reproduce &nbsp;&rarr;&nbsp; <code>python3 scripts/09-stage3c-capex-by-technology.py</code><br>
  input: <code>harmonized-costs.csv</code>
  &nbsp;&middot;&nbsp; outputs: <code>figures/capex-by-technology.png</code>,
  <code>outputs/analysis/capex-by-technology.csv</code>
</footer>
"""


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("costs", nargs="?", default=str(DEFAULT_COSTS))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--figures-dir", default=str(DEFAULT_FIGDIR))
    ap.add_argument("--doc", default=str(DEFAULT_DOC))
    args = ap.parse_args()

    points, kept, order, meta = build(load_csv(Path(args.costs)))
    if not order:
        raise SystemExit("no technology cleared the coverage threshold")

    stats = compute_stats(points, kept, order, meta)
    write_csv(Path(args.outdir) / "capex-by-technology.csv",
              [p for p in points if p["technology"] in kept])

    fig_path = Path(args.figures_dir) / "capex-by-technology.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _style()
    fig_capex(kept, order, stats, fig_path)

    png_b64 = base64.b64encode(fig_path.read_bytes()).decode()
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_doc(stats, png_b64), encoding="utf-8")

    print(f"USD2025/MW rows          : {stats['n_mw_rows']}")
    print(f"  O&M / equipment dropped : {stats['dropped_detail']}")
    print(f"  unclassified            : {stats['unclassified']}")
    print(f"  plotted                 : {stats['n_points']} values, "
          f"{stats['n_tech']} technologies, {stats['n_papers']} papers")
    for t in reversed(order):
        d = stats["per_tech"][t]
        print(f"    {t:22} median ${d['median']:.2f}M/MW  "
              f"({d['min']:.2f}-{d['max']:.2f}, n={d['n']}/{d['n_papers']}p)")
    print(f"\nfigure -> {fig_path}\ndoc    -> {doc_path}")


if __name__ == "__main__":
    main()
