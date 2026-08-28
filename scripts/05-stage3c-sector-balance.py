# scripts/stage3d-sector-balance.py  (stage 3d -- exploratory, runs after 3c)
#
# Exploratory analysis: how do sectors contribute to the net-zero balance?
#
# Net-zero scenarios are taken from netzero-flags.csv (written by
# scripts/04-stage3c-harmonize.py).  For each flagged scenario, at its last reported year,
# every sector's net emissions are expressed as a share of the scenario's GROSS
# positive (residual) emissions that year -- so positive sectors sum to +100%,
# removals are negative, and a 10 GtCO2 scenario sits on the same axis as a
# 4 MtCO2 one.
#
# Several sub-sectors can share the same Sector_std (e.g. Energy = Power +
# Transport + Buildings).  They are aggregated:
#   - values within one Subsector_std      -> median  (duplicate / conflicting extractions)
#   - across distinct Subsector_std        -> sum
#   - if a sector-total row (blank Subsector_std) exists -> use it instead
#
# INPUT   outputs/output-stage3c/harmonized-emissions.csv
#         outputs/output-stage3c/netzero-flags.csv
# OUTPUT  outputs/analysis/sector-balance-at-netzero.csv
#         outputs/analysis/sector-balance-scenarios.csv
#         outputs/analysis/sector-balance-summary.csv
#         figures/sector-balance-strip.png
#         figures/sector-balance-scenarios.png
#
#   python3 scripts/stage3d-sector-balance.py     (run from the repo root)

from pathlib import Path
import argparse
import csv
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_common import canon_scenario, canon_area   # noqa: E402


SECTOR_ORDER = ["Energy", "Industry", "Agriculture", "Waste", "LULUCF", "Other"]

SECTOR_COLOR = {
    "Energy":      "#2a78d6",
    "Industry":    "#eb6834",
    "Agriculture": "#1baf7a",
    "Waste":       "#eda100",
    "LULUCF":      "#e87ba4",
    "Other":       "#4a3aa7",
}
SOURCE_C, SINK_C, NEAR0_C = "#d1382f", "#2a78d6", "#8a8a84"
INK, INK2, MUTED, GRID = "#1a1a1a", "#55534e", "#8a8a84", "#dedcd5"

ABSOLUTE_UNITS = {"MtCO2", "MtCO2/yr", "MtCO2eq", "MtCO2eq/yr"}

CAVEAT = (
    "Exploratory. Most net-zero scenarios report only an economy-wide total — "
    "very few also break emissions out by sector at the neutrality year. "
    "CO₂ and CO₂e are pooled; national and subnational are mixed. "
    "A LULUCF share below −100% means removals exceed the captured residual "
    "(net-negative scenario, or not every emitting sector reported)."
)

DEFAULT_EMISSIONS = Path("outputs/output-stage3c/harmonized-emissions.csv")
DEFAULT_FLAGS = Path("outputs/output-stage3c/netzero-flags.csv")
DEFAULT_OUTDIR = Path("outputs/analysis")
DEFAULT_FIGDIR = Path("figures")


# ============================================================================
# LOADING
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


def geographic_scope(area):
    a = (area or "").strip().lower()
    if not a:
        return "unspecified"
    if re.search(r"\b(global|world|worldwide)\b", a):
        return "global"
    if re.search(r"\b(plant|farm|facility|site|refinery|mine|unit|turbine|"
                 r"boiler|kiln|estate)\b", a):
        return "facility"
    if re.search(r"\b(province|state|voivodeship|region|county|city|prefecture|"
                 r"bay area|greater|municipal|canton)\b", a):
        return "subnational"
    return "national"


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def load_netzero_flags(path):
    """{(paper_id, scenario_canon, area_canon, basis): {...}} for is_netzero."""
    flags = {}
    for r in load_csv(path):
        if r.get("is_netzero") != "True":
            continue
        key = (r["paper_id"],
               (r.get("scenario_canon") or canon_scenario(r.get("Scenario_Name"))),
               (r.get("area_canon") or canon_area(r.get("Area"))),
               r.get("basis", ""))
        fy = to_float(r.get("first_year"))
        flags[key] = {
            "last_year": int(float(r["last_year"])),
            "first_year": int(fy) if fy is not None else None,
            "first_emissions": to_float(r.get("first_emissions_Mt")),
            "last_emissions": to_float(r.get("last_emissions_Mt")),
            "net_negative": r.get("is_net_negative") == "True",
            "fraction": to_float(r.get("netzero_fraction")),
        }
    return flags


# ============================================================================
# SECTOR AGGREGATION
# ============================================================================

def source_ref(row):
    """'Figure 3' / 'Table S5' / 'Section 4.2' reference for one harmonised row,
    so a data point can be traced back to the paper for manual checking."""
    st = (row.get("Source_Type") or "").strip()
    sn = (row.get("Source_Number") or "").strip()
    if st and sn:
        return f"{st} {sn}"
    return st or sn or "Not reported"


def aggregate_sectors(rows_at_year):
    """rows_at_year: list of dicts with sector, subsector, value, src, raw.
    Return {sector: (aggregated_value, [contributing rows])}."""
    by_sector = {}
    for r in rows_at_year:
        by_sector.setdefault(r["sector"], {}).setdefault(
            r["subsector"], []
        ).append(r)

    out = {}
    for sector, by_sub in by_sector.items():
        if "" in by_sub:                       # an explicit sector-total row
            used = by_sub[""]
            value = statistics.median(x["value"] for x in used)
        else:                                  # sum the distinct sub-sectors
            used = [x for vs in by_sub.values() for x in vs]
            value = sum(statistics.median(x["value"] for x in vs)
                        for vs in by_sub.values())
        out[sector] = (value, used)
    return out


# ============================================================================
# BUILD
# ============================================================================

def build(emission_rows, flags):
    # Area / Publication_Year / etc. blanks are already back-filled from
    # sibling rows upstream in scripts/04-stage3c-harmonize.py (Context_Filled)
    clean = []
    for r in emission_rows:
        value = to_float(r.get("Value_std"))
        unit = r.get("Value_std_Unit", "")
        year = parse_year(r.get("Year"))
        if value is None or unit not in ABSOLUTE_UNITS or year is None:
            continue
        if geographic_scope(r.get("Area")) == "facility":
            continue
        area = (r.get("Area") or "").strip()
        clean.append({
            "paper_id": r["paper_id"],
            "title": r.get("Title", ""),
            "doi": r.get("DOI", ""),
            "scenario": (r.get("Scenario_Name") or "").strip(),
            "area": area,
            "scen_c": (r.get("Scenario_canon")
                       or canon_scenario(r.get("Scenario_Name"))),
            "area_c": (r.get("Area_canon") or canon_area(area)),
            "basis": "CO2eq" if "eq" in unit else "CO2",
            "sector": (r.get("Sector_std") or "").strip(),
            "subsector": (r.get("Subsector_std") or "").strip(),
            "year": year,
            "value": value,
            "raw": f"{(r.get('Value') or '').strip()} "
                   f"{(r.get('Value_Unit') or '').strip()}".strip(),
            "src": source_ref(r),
        })

    groups = {}
    for r in clean:
        groups.setdefault(
            (r["paper_id"], r["scen_c"], r["area_c"], r["basis"]), []
        ).append(r)

    tidy, scenario_rows = [], []

    for key, recs in sorted(groups.items()):
        flag = flags.get(key)
        if flag is None:
            continue

        netzero_year = flag["last_year"]
        at_year = [
            x for x in recs
            if x["year"] == netzero_year
            and x["sector"] in SECTOR_ORDER          # exclude "All sectors"
        ]
        per_sector = aggregate_sectors(at_year)
        if len(per_sector) < 2:
            continue

        gross_positive = sum(v for v, _ in per_sector.values() if v > 0)
        if gross_positive <= 0:
            continue

        # the net-zero flag can be set from the economy-wide "All sectors"
        # total, while this figure decomposes the reported sectors.  When the
        # sectors actually plotted sum to net-POSITIVE above the flag's
        # threshold, the scenario is not net zero as drawn -- drop it so the
        # figure stays honest (e.g. p282: total hits 0 in 2050 but the
        # reported sectors sum to +1 MtCO2).  Net-negative sums pass, as they
        # do in the flag itself.
        net_last = sum(v for v, _ in per_sector.values())
        base = flag["first_emissions"]
        frac = flag["fraction"]
        if frac is not None and base and base > 0 and net_last > frac * base:
            continue

        paper_id, _scen_c, _area_c, basis = key
        # representative display strings for this canonical group
        scenario = max({r["scenario"] for r in recs},
                       key=lambda s: [r["scenario"] for r in recs].count(s))
        area = max({r["area"] for r in recs},
                   key=lambda a: [r["area"] for r in recs].count(a))
        scope = geographic_scope(area)
        meta = recs[0]

        for sector, (value, used) in per_sector.items():
            tidy.append({
                "paper_id": paper_id, "scenario": scenario, "area": area,
                "scope": scope, "basis": basis, "netzero_year": netzero_year,
                "net_negative": flag["net_negative"],
                "n_sectors": len(per_sector), "sector": sector,
                "emissions_MtCO2": round(value, 3),
                "gross_positive_MtCO2": round(gross_positive, 3),
                "share_of_gross": round(value / gross_positive, 4),
                "n_source_rows": len(used),
                "raw_values": " | ".join(sorted({x["raw"] for x in used})),
                "source": " | ".join(sorted({x["src"] for x in used})),
            })

        scenario_rows.append({
            "paper_id": paper_id, "scenario": scenario, "area": area,
            "scope": scope, "basis": basis, "netzero_year": netzero_year,
            "net_negative": flag["net_negative"], "n_sectors": len(per_sector),
            "sectors": "|".join(sorted(per_sector)),
            "gross_positive_MtCO2": round(gross_positive, 3),
            "title": meta["title"], "doi": meta["doi"],
        })

    return tidy, scenario_rows


def summarize(tidy):
    out = []
    for sector in SECTOR_ORDER:
        recs = [r for r in tidy if r["sector"] == sector]
        vals = sorted(r["share_of_gross"] for r in recs)
        if not vals:
            continue
        if len(vals) >= 4:
            q1, _, q3 = statistics.quantiles(vals, n=4)
        else:
            q1, q3 = vals[0], vals[-1]
        out.append({
            "sector": sector, "n_scenarios": len(vals),
            "n_papers": len({r["paper_id"] for r in recs}),
            "n_positive": sum(1 for v in vals if v > 0.01),
            "n_negative": sum(1 for v in vals if v < -0.01),
            "min": round(vals[0], 4), "q1": round(q1, 4),
            "median": round(statistics.median(vals), 4),
            "q3": round(q3, 4), "max": round(vals[-1], 4),
        })
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ============================================================================
# FIGURES
# ============================================================================

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": GRID, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 160, "svg.fonttype": "none",
    })


def _jitter(n, spread=0.30):
    if n == 1:
        return [0.0]
    step = (2 * spread) / (n - 1)
    return [-spread + i * step for i in range(n)]


def fig_strip(tidy, summary, path, n_netzero):
    order = [s["sector"] for s in sorted(summary, key=lambda s: -s["median"])]
    xmin, xmax = -2.6, 1.15

    fig, ax = plt.subplots(figsize=(8.4, 0.72 * len(order) + 1.4))
    ax.axvspan(0, xmax, color=SOURCE_C, alpha=0.045, lw=0)
    ax.axvspan(xmin, 0, color=SINK_C, alpha=0.045, lw=0)
    ax.axvline(0, color="#b4b2aa", lw=1.4, zorder=2)

    for i, sector in enumerate(order):
        y = len(order) - 1 - i
        vals = sorted(r["share_of_gross"] for r in tidy if r["sector"] == sector)
        recs = [r for r in tidy if r["sector"] == sector]
        if not vals:
            continue

        # dots on top of the row band, stats bar just below
        clipped_lo = []
        for dy, r in zip(_jitter(len(recs)), recs):
            share = r["share_of_gross"]
            c = SOURCE_C if share > 0.03 else SINK_C if share < -0.03 else NEAR0_C
            xs = min(max(share, xmin), xmax)
            off = share < xmin or share > xmax
            ax.scatter([xs], [y + 0.08 + dy], s=34, c=c,
                       marker="<" if share < xmin else ">" if share > xmax else "o",
                       edgecolors="white", linewidths=0.7,
                       alpha=0.4 if r["n_sectors"] == 2 else 0.95, zorder=5)
            if share < xmin:
                clipped_lo.append(share)
            elif share > xmax:
                ax.annotate(f"{share * 100:+.0f}%", (xmax, y + 0.08 + dy),
                            fontsize=7, color=MUTED, va="center", ha="right")
        # one aggregated label for the (often several) points past the left edge
        if clipped_lo:
            ax.annotate(
                f"{len(clipped_lo)} past axis, to {min(clipped_lo) * 100:+.0f}%",
                (xmin + 0.03, y + 0.30), fontsize=6.8, color=MUTED,
                va="center", ha="left")

        med = statistics.median(vals)
        if len(vals) >= 4:
            q1, _, q3 = statistics.quantiles(vals, n=4)
            ax.plot([q1, q3], [y - 0.30, y - 0.30], color=INK2, lw=1.4, zorder=3)
        ax.plot([med, med], [y - 0.38, y - 0.22], color=INK, lw=2.6, zorder=4)
        ax.text(xmax + 0.04, y + 0.02, f"{med * 100:+.0f}%", va="center",
                ha="left", fontsize=8.5, fontweight="bold", color=INK)

        ax.text(xmin - 0.05, y + 0.06,
                "Other / mixed" if sector == "Other" else sector,
                va="center", ha="right", fontsize=10, color=INK)
        n_papers = len({r["paper_id"] for r in recs})
        ax.text(xmin - 0.05, y - 0.30, f"n={len(vals)} ({n_papers})",
                va="center", ha="right", fontsize=7.5, color=MUTED)
        if i:
            ax.axhline(y + 0.5, color=GRID, lw=0.8, zorder=1)

    ax.set_xlim(xmin, xmax + 0.24)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks([])
    ax.set_xticks([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([f"{t * 100:+.0f}%".replace("+0%", "0%")
                        for t in ax.get_xticks()])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("← removal / sink        share of gross residual emissions"
                  "        residual emissions →", fontsize=8.5)

    n_sc = len({(r["paper_id"], r["scenario"], r["area"], r["basis"])
                for r in tidy})
    ax.set_title(f"Sector balance at net zero  ·  {n_sc} scenarios, "
                 f"{len({r['paper_id'] for r in tidy})} papers, "
                 f"{n_netzero} flagged net zero in total\n"
                 "one dot per scenario  ·  bar = median  ·  whisker = IQR"
                 "  ·  faded = 2-sector only  ·  n = scenarios (papers)",
                 fontsize=10.5, loc="left", color=INK, pad=12)
    fig.text(0.5, 0.005, CAVEAT, ha="center", va="bottom",
             fontsize=7.3, color=MUTED, wrap=True)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _scenario_key(r):
    return (r["paper_id"], r["scenario"], r["area"], r["basis"])


def _scenario_label(k):
    paper_id, scenario, area, basis = k
    sc = scenario if len(scenario) <= 24 else scenario[:23] + "…"
    a = area if len(area) <= 15 else area[:14] + "…"
    b = "CO₂e" if basis == "CO2eq" else "CO₂"
    return f"p{paper_id} · {a} · {sc} ({b})"


def fig_strip_by_scenario(tidy, summary, path, n_netzero):
    """The sector-balance strip, but every dot is coloured by its scenario
    (legend on the right) so a single pathway can be picked out row to row.
    A faint line links each scenario's dots as a tracing aid."""
    order = [s["sector"] for s in sorted(summary, key=lambda s: -s["median"])]
    xmin, xmax = -2.6, 1.15
    row_y = {sec: len(order) - 1 - i for i, sec in enumerate(order)}

    keys = sorted({_scenario_key(r) for r in tidy})
    palette = [plt.cm.tab20(i / 20) for i in range(20)]
    colour = {k: palette[i % 20] for i, k in enumerate(keys)}

    fig, ax = plt.subplots(figsize=(10.6, 0.82 * len(order) + 1.8))
    ax.axvspan(0, xmax, color=SOURCE_C, alpha=0.04, lw=0)
    ax.axvspan(xmin, 0, color=SINK_C, alpha=0.04, lw=0)
    ax.axvline(0, color="#b4b2aa", lw=1.4, zorder=2)

    # deterministic dot slot per (sector, scenario): spread scenarios that
    # appear in the row evenly across the band, keyed by global scenario order
    by_sector = {sec: [r for r in tidy if r["sector"] == sec] for sec in order}
    slot = {}
    for sec, recs in by_sector.items():
        ks = sorted({_scenario_key(r) for r in recs}, key=keys.index)
        for dy, k in zip(_jitter(len(ks), spread=0.34), ks):
            slot[(sec, k)] = dy

    for i, sector in enumerate(order):
        y = row_y[sector]
        recs = by_sector[sector]
        vals = sorted(r["share_of_gross"] for r in recs)
        if not vals:
            continue
        if len(vals) >= 4:
            q1, _, q3 = statistics.quantiles(vals, n=4)
            ax.plot([q1, q3], [y - 0.34, y - 0.34], color=INK2, lw=1.3, zorder=3)
        ax.plot([statistics.median(vals)] * 2, [y - 0.42, y - 0.26],
                color=INK, lw=2.4, zorder=4)
        ax.text(xmax + 0.04, y, f"{statistics.median(vals) * 100:+.0f}%",
                va="center", ha="left", fontsize=8.5, fontweight="bold",
                color=INK)
        ax.text(xmin - 0.05, y + 0.05,
                "Other / mixed" if sector == "Other" else sector,
                va="center", ha="right", fontsize=10, color=INK)
        ax.text(xmin - 0.05, y - 0.32, f"n={len(vals)} "
                f"({len({r['paper_id'] for r in recs})})",
                va="center", ha="right", fontsize=7.5, color=MUTED)
        if i:
            ax.axhline(y + 0.5, color=GRID, lw=0.8, zorder=1)

    by_scen = {}
    for r in tidy:
        by_scen.setdefault(_scenario_key(r), []).append(r)
    clipped = {}
    for k in keys:
        rs = sorted(by_scen[k], key=lambda r: order.index(r["sector"]))
        pts = []
        for r in rs:
            share = r["share_of_gross"]
            xs = min(max(share, xmin), xmax)
            pts.append((xs, row_y[r["sector"]] + 0.08 + slot[(r["sector"], k)],
                        share))
            if share < xmin:
                clipped[r["sector"]] = clipped.get(r["sector"], 0) + 1
        c = colour[k]
        if len(pts) >= 2:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=c, lw=0.6, alpha=0.18, zorder=5)
        for xs, yy, share in pts:
            m = "<" if share < xmin else ">" if share > xmax else "o"
            ax.scatter([xs], [yy], s=36, color=c, marker=m,
                       edgecolors="white", linewidths=0.6, zorder=6)

    for sector, nclip in clipped.items():
        ax.annotate(f"{nclip} past axis", (xmin + 0.03, row_y[sector] + 0.34),
                    fontsize=6.8, color=MUTED, va="center", ha="left")

    ax.set_xlim(xmin, xmax + 0.24)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks([])
    ax.set_xticks([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([f"{t * 100:+.0f}%".replace("+0%", "0%")
                        for t in ax.get_xticks()])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("← removal / sink        share of gross residual emissions"
                  "        residual emissions →", fontsize=8.5)

    ax.set_title(f"Sector balance at net zero, by scenario  ·  {len(keys)} "
                 f"scenarios, {len({r['paper_id'] for r in tidy})} papers, "
                 f"{n_netzero} flagged net zero in total\n"
                 "one colour per scenario  ·  faint line links a scenario "
                 "across sectors  ·  grey bar = median, whisker = IQR",
                 fontsize=10.5, loc="left", color=INK, pad=12)

    handles = [plt.Line2D([0], [0], marker="o", ls="", color=colour[k],
                          mec="white", ms=8) for k in keys]
    ax.legend(handles, [_scenario_label(k) for k in keys],
              loc="center left", bbox_to_anchor=(1.015, 0.5), frameon=False,
              fontsize=7.2, labelspacing=.55, handletextpad=.4,
              title=f"scenario  ·  {len(keys)} colours"
                    + ("  (tab20 repeats past 20)" if len(keys) > 20 else ""),
              title_fontsize=7.5, alignment="left")

    fig.tight_layout(rect=(0, 0.02, 0.995, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_scenarios(tidy, scenario_rows, path):
    scen = sorted(scenario_rows,
                  key=lambda s: (s["netzero_year"], s["area"], s["paper_id"]))
    by_key = {}
    for r in tidy:
        by_key.setdefault(
            (r["paper_id"], r["scenario"], r["area"], r["basis"]), {}
        )[r["sector"]] = r["share_of_gross"]

    xmin, xmax = -3.3, 1.3
    fig_h = 0.34 * len(scen) + 2.0
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    bottom_frac = min(0.16, 0.75 / fig_h)   # room for the legend under the axes

    for row, s in enumerate(scen):
        y = len(scen) - 1 - row
        shares = by_key[(s["paper_id"], s["scenario"], s["area"], s["basis"])]
        pos = [(sec, shares[sec]) for sec in SECTOR_ORDER
               if shares.get(sec, 0) > 0]
        neg = [(sec, shares[sec]) for sec in SECTOR_ORDER
               if shares.get(sec, 0) < 0]
        acc = 0.0
        for sec, v in pos:
            ax.barh(y, v, left=acc, height=0.66, color=SECTOR_COLOR[sec],
                    edgecolor="white", linewidth=0.6)
            acc += v
        acc = 0.0
        for sec, v in neg:
            ax.barh(y, v, left=acc, height=0.66, color=SECTOR_COLOR[sec],
                    edgecolor="white", linewidth=0.6)
            acc += v
        # the axis stops at xmin; a bar that runs past it is chopped by the
        # plot edge with no cue -- label its true total so it is not read as
        # bottoming out at the axis (e.g. p436 removals reach -1133%)
        if acc < xmin:
            ax.annotate(f"{acc * 100:+.0f}%", (xmin, y), xytext=(3, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=6.8, fontweight="bold", color="white",
                        zorder=6)
        label = s["area"] if len(s["area"]) <= 26 else s["area"][:25] + "…"
        ax.text(-0.02 if xmin > -0.02 else xmin - 0.04, y,
                f"{label}\np{s['paper_id']} · {s['netzero_year']} · "
                f"{'CO₂e' if s['basis'] == 'CO2eq' else 'CO₂'}",
                va="center", ha="right", fontsize=7.2, color=INK2,
                linespacing=1.25)

    ax.axvline(0, color="#b4b2aa", lw=1.4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, len(scen) - 0.3)
    ax.set_yticks([])
    ax.set_xticks([-3, -2, -1, 0, 1])
    ax.set_xticklabels(["-300%", "-200%", "-100%", "0%", "+100%"])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")

    ax.set_title("Every net-zero scenario, end to end  ·  "
                 "bar normalised so residual emissions = 100%, "
                 "removals stacked left",
                 fontsize=10.5, loc="left", color=INK, pad=22)

    used = [s for s in SECTOR_ORDER
            if any(v.get(s) for v in by_key.values())]
    handles = [plt.Rectangle((0, 0), 1, 1, color=SECTOR_COLOR[s]) for s in used]
    fig.legend(handles,
               ["Other / mixed" if s == "Other" else s for s in used],
               loc="lower center", ncol=len(used), frameon=False, fontsize=8,
               handlelength=1.1, columnspacing=1.3,
               bbox_to_anchor=(0.5, bottom_frac * 0.32))
    fig.text(0.5, -0.02, CAVEAT, ha="center", va="top", fontsize=7.3,
             color=MUTED, wrap=True)

    fig.tight_layout(rect=(0, bottom_frac, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emissions", nargs="?", default=str(DEFAULT_EMISSIONS))
    ap.add_argument("--flags", default=str(DEFAULT_FLAGS))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--figures-dir", default=str(DEFAULT_FIGDIR))
    args = ap.parse_args()

    flags = load_netzero_flags(Path(args.flags))
    tidy, scenario_rows = build(load_csv(Path(args.emissions)), flags)
    summary = summarize(tidy)

    outdir = Path(args.outdir)
    write_csv(outdir / "sector-balance-at-netzero.csv", tidy, [
        "paper_id", "scenario", "area", "scope", "basis", "netzero_year",
        "net_negative", "n_sectors", "sector", "emissions_MtCO2",
        "gross_positive_MtCO2", "share_of_gross",
        "n_source_rows", "raw_values", "source",
    ])
    write_csv(outdir / "sector-balance-scenarios.csv", scenario_rows, [
        "paper_id", "scenario", "area", "scope", "basis", "netzero_year",
        "net_negative", "n_sectors", "sectors", "gross_positive_MtCO2",
        "title", "doi",
    ])
    write_csv(outdir / "sector-balance-summary.csv", summary, [
        "sector", "n_scenarios", "n_papers", "n_positive", "n_negative",
        "min", "q1", "median", "q3", "max",
    ])

    figdir = Path(args.figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)
    _style()
    if scenario_rows:
        fig_strip(tidy, summary, figdir / "sector-balance-strip.png", len(flags))
        fig_strip_by_scenario(
            tidy, summary,
            figdir / "sector-balance-strip-by-scenario.png", len(flags))
        fig_scenarios(tidy, scenario_rows,
                      figdir / "sector-balance-scenarios.png")
    else:
        print("no scenarios to plot")

    print(f"net-zero scenarios flagged           : {len(flags)}")
    print(f"  ... with >=2 sectors at last year  : {len(scenario_rows)}")
    print(f"  by basis  : "
          + ", ".join(f"{k}={sum(1 for s in scenario_rows if s['basis'] == k)}"
                      for k in ("CO2", "CO2eq")))
    print(f"  by scope  : "
          + ", ".join(f"{k}={sum(1 for s in scenario_rows if s['scope'] == k)}"
                      for k in ("global", "national", "subnational", "unspecified")))
    print(f"  n_sectors : "
          + ", ".join(f"{k}={sum(1 for s in scenario_rows if s['n_sectors'] == k)}"
                      for k in (2, 3, 4, 5, 6)))
    print()
    print(f"{'sector':<13}{'scen':>5}{'pap':>5}{'  n+':>5}{'  n-':>5}"
          f"{'median share':>14}")
    for s in summary:
        print(f"{s['sector']:<13}{s['n_scenarios']:>5}{s['n_papers']:>5}"
              f"{s['n_positive']:>5}{s['n_negative']:>5}"
              f"{s['median'] * 100:>12.0f} %")
    print(f"\nfigures -> {figdir}/sector-balance-strip.png, "
          f"{figdir}/sector-balance-strip-by-scenario.png, "
          f"{figdir}/sector-balance-scenarios.png")


if __name__ == "__main__":
    main()
