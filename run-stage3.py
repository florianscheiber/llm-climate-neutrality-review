# run-stage3.py
#
# One entry point that runs the stage-3 extraction pipeline end to end, in
# the correct order. Each step is a separate script (documented at the top
# of its own file); this just chains them and stops on the first failure.
#
# PIPELINE  (each step is scripts/<name>, run from the repo root)
#
#   markdown         scripts/pdf-to-markdown-extraction.py
#                      inputs/fulltext_pdf/included-after-stage-3-100/*.pdf
#                        -> intermediate/pdf-extraction/<id>/  (markdown + figure images)
#
#   extract          scripts/01-stage3c-data-extraction.py          [needs LLM_ROUTER_API_KEY]
#                      intermediate/pdf-extraction/
#                        -> outputs/output-stage3c/<id>/  (mitigation/costs/emissions CSV
#                           + <id>-candidates.txt for figures it could not read)
#
#   candidates       scripts/02-stage3c-candidates-extraction.py    [needs LLM_ROUTER_API_KEY]
#                      re-reads the flagged figures from full-page 300-DPI renders
#                        -> outputs/output-stage3c-candidates/<id>/
#
#   merge            scripts/03-stage3c-merge-extractions.py
#                      per-paper CSVs (both passes) + screening metadata
#                        -> outputs/output-stage3c/merged-{mitigation,costs,emissions}.csv
#
#   harmonize        scripts/04-stage3c-harmonize.py                [World Bank API, then cached]
#                      merged-*.csv
#                        -> outputs/output-stage3c/harmonized-*.csv
#                        -> outputs/output-stage3c/netzero-flags.csv
#
#   --- exploratory analyses over the harmonized tables (05-10) ---
#   each reads harmonized-*.csv / netzero-flags.csv and writes
#   outputs/analysis/*.csv, figures/*.png and its own docs/*.html:
#
#   sector-balance            scripts/05-stage3c-sector-balance.py
#   carbon-price              scripts/06-stage3c-carbon-price.py
#   mitigation-contribution   scripts/07-stage3c-mitigation-contribution.py
#   electricity-price         scripts/08-stage3c-electricity-price.py
#   capex                     scripts/09-stage3c-capex-by-technology.py
#   macro-economic-cost       scripts/10-stage3c-macro-economic-cost.py
#
# USAGE
#
#   # the whole pipeline, markdown -> 09:
#   export LLM_ROUTER_API_KEY=<key>
#   python3 run-stage3.py
#
#   # preview the commands without running anything:
#   python3 run-stage3.py --dry-run
#
#   # the markdown + first extraction already exist, just do the rest:
#   python3 run-stage3.py --from candidates
#
#   # stop after the harmonized tables (skip the 05-09 analyses):
#   python3 run-stage3.py --to harmonize
#
#   # re-run just the analyses (harmonized tables already exist):
#   python3 run-stage3.py --from sector-balance
#
#   # re-merge and re-harmonize only:
#   python3 run-stage3.py --only merge,harmonize
#
#   # dry-run the candidate locator (no API calls), stop there:
#   python3 run-stage3.py --only candidates --candidates-dry-run
#
#   # test on the first 3 papers:
#   python3 run-stage3.py --limit 3
#
# NOTES
#
# - `extract` and `candidates` skip papers that are already done, so a
#   re-run is cheap and an interrupted run resumes on its own.
# - If a step fails, the run stops and prints the exact command to resume
#   from that step.

from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys
import time


# ============================================================================
# STEP DEFINITIONS
# ============================================================================

# name -> (script path relative to repo root, needs_api_key)
STEPS = [
    ("markdown",       "scripts/pdf-to-markdown-extraction.py",       False),
    ("extract",        "scripts/01-stage3c-data-extraction.py",       True),
    ("candidates",     "scripts/02-stage3c-candidates-extraction.py", True),
    ("merge",          "scripts/03-stage3c-merge-extractions.py",     False),
    ("harmonize",      "scripts/04-stage3c-harmonize.py",             False),
    ("sector-balance", "scripts/05-stage3c-sector-balance.py",        False),
    ("carbon-price",   "scripts/06-stage3c-carbon-price.py",          False),
    ("mitigation-contribution",
     "scripts/07-stage3c-mitigation-contribution.py",                 False),
    ("electricity-price",
     "scripts/08-stage3c-electricity-price.py",                       False),
    ("capex",          "scripts/09-stage3c-capex-by-technology.py",   False),
    ("macro-economic-cost",
     "scripts/10-stage3c-macro-economic-cost.py",                     False),
]
STEP_NAMES = [name for name, _, _ in STEPS]

# 05-10: standalone analyses over the harmonized tables, run with no arguments
ANALYSIS_STEPS = {
    "sector-balance", "carbon-price", "mitigation-contribution",
    "electricity-price", "capex", "macro-economic-cost",
}

# steps run when neither --to nor --only is given (the whole pipeline)
DEFAULT_LAST_STEP = STEP_NAMES[-1]

MARKDOWN_DIR = "intermediate/pdf-extraction"
STAGE3C_DIR = "outputs/output-stage3c"
CANDIDATES_DIR = "outputs/output-stage3c-candidates"
DEFAULT_PDF_DIR = "inputs/fulltext_pdf/included-after-stage-3-100"


# ============================================================================
# COMMAND BUILDERS
# ============================================================================

def build_command(step_name, args, python):

    script = dict((n, s) for n, s, _ in STEPS)[step_name]
    cmd = [python, script]

    if step_name == "markdown":
        cmd += [args.pdf_dir, MARKDOWN_DIR]
        if args.dpi is not None:
            cmd += ["--dpi", str(args.dpi)]

    elif step_name == "extract":
        cmd += [MARKDOWN_DIR, STAGE3C_DIR]
        if args.model:
            cmd += ["--model", args.model]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]

    elif step_name == "candidates":
        cmd += [
            "--stage3c-dir", STAGE3C_DIR,
            "--pdf-dir", args.pdf_dir,
            "--output-dir", CANDIDATES_DIR,
        ]
        if args.model:
            cmd += ["--model", args.model]
        if args.candidates_papers:
            cmd += ["--papers", args.candidates_papers]
        if args.candidates_all:
            cmd += ["--all"]
        if args.candidates_force:
            cmd += ["--force"]
        if args.candidates_dry_run:
            cmd += ["--dry-run"]

    elif step_name == "merge":
        cmd += [STAGE3C_DIR, STAGE3C_DIR]
        if not args.no_candidates_merge and Path(CANDIDATES_DIR).is_dir():
            cmd += ["--extra-input-dir", CANDIDATES_DIR]

    elif step_name == "harmonize":
        cmd += [STAGE3C_DIR, STAGE3C_DIR]
        if args.netzero_fraction is not None:
            cmd += ["--netzero-fraction", str(args.netzero_fraction)]
        if args.refresh:
            cmd += ["--refresh"]

    elif step_name in ANALYSIS_STEPS:
        pass  # 05-09 run with no arguments (all defaults)

    return cmd


def step_needs_key(step_name, args):
    needs = dict((n, k) for n, _, k in STEPS)[step_name]
    if step_name == "candidates" and args.candidates_dry_run:
        return False
    return needs


# ============================================================================
# STEP SELECTION
# ============================================================================

def resolve_selection(args):

    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        for s in wanted:
            if s not in STEP_NAMES:
                sys.exit(f"unknown step in --only: {s}\nknown: {', '.join(STEP_NAMES)}")
        selected = [s for s in STEP_NAMES if s in wanted]
    else:
        start = args.from_step or STEP_NAMES[0]
        end = args.to_step or DEFAULT_LAST_STEP
        for label, s in (("--from", start), ("--to", end)):
            if s not in STEP_NAMES:
                sys.exit(f"unknown step in {label}: {s}\nknown: {', '.join(STEP_NAMES)}")
        i, j = STEP_NAMES.index(start), STEP_NAMES.index(end)
        if i > j:
            sys.exit(f"--from ({start}) is after --to ({end})")
        selected = STEP_NAMES[i:j + 1]

    skip = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
    for s in skip:
        if s not in STEP_NAMES:
            sys.exit(f"unknown step in --skip: {s}")
    selected = [s for s in selected if s not in skip]

    if not selected:
        sys.exit("nothing to run after applying --skip")

    return selected


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Run the stage-3 extraction pipeline in order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sel = parser.add_argument_group("step selection")
    sel.add_argument("--from", dest="from_step", metavar="STEP",
                     help=f"start at STEP (default: {STEP_NAMES[0]})")
    sel.add_argument("--to", dest="to_step", metavar="STEP",
                     help=f"stop after STEP (default: {DEFAULT_LAST_STEP})")
    sel.add_argument("--only", metavar="STEP[,STEP]",
                     help="run only these steps")
    sel.add_argument("--skip", metavar="STEP[,STEP]",
                     help="skip these steps")

    pt = parser.add_argument_group("pass-through options")
    pt.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR,
                    help=f"source PDF folder (default: {DEFAULT_PDF_DIR})")
    pt.add_argument("--dpi", type=int, help="markdown: figure-region raster DPI")
    pt.add_argument("--limit", type=int,
                    help="extract: process only the first N papers")
    pt.add_argument("--model", help="extract / candidates: model id override")
    pt.add_argument("--candidates-dry-run", action="store_true",
                    help="candidates: locate figures only, no API calls")
    pt.add_argument("--candidates-all", action="store_true",
                    help="candidates: also attempt un-digitisable figures")
    pt.add_argument("--candidates-papers", metavar="LIST",
                    help="candidates: comma-separated folder ids")
    pt.add_argument("--candidates-force", action="store_true",
                    help="candidates: reprocess papers that already have output")
    pt.add_argument("--no-candidates-merge", action="store_true",
                    help="merge: do NOT fold the candidates pass into merged-*.csv")
    pt.add_argument("--netzero-fraction", type=float,
                    help="harmonize: net-zero threshold (default 0.05)")
    pt.add_argument("--refresh", action="store_true",
                    help="harmonize: re-download World Bank reference data")

    parser.add_argument("--python", default=sys.executable,
                        help="interpreter to run each step with")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands, run nothing")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    selected = resolve_selection(args)

    # fail fast if a pass-through option was given but the step that actually
    # consumes it is not in the selected range -- otherwise the option is
    # silently ignored and the run produces a stale/wrong result (e.g.
    # `--from sector-balance --netzero-fraction 0` never re-flags anything)
    OPTION_STEP = [
        (args.netzero_fraction is not None, "--netzero-fraction", "harmonize"),
        (args.refresh, "--refresh", "harmonize"),
        (args.limit is not None, "--limit", "extract"),
        (args.dpi is not None, "--dpi", "markdown"),
    ]
    stranded = [(opt, step) for given, opt, step in OPTION_STEP
                if given and step not in selected]
    if stranded:
        lines = "\n".join(f"  {opt} is consumed by the '{step}' step, "
                          f"which is not in this run" for opt, step in stranded)
        need = min((s for _, s in stranded), key=STEP_NAMES.index)
        sys.exit(f"option(s) would be ignored:\n{lines}\n"
                 f"  re-run starting no later than that step, e.g. "
                 f"--from {need}")

    # fail fast if an API-key step is selected but the key is missing
    if not args.dry_run:
        missing_key = [s for s in selected if step_needs_key(s, args)]
        if missing_key and not os.environ.get("LLM_ROUTER_API_KEY"):
            sys.exit(
                "LLM_ROUTER_API_KEY is not set, needed for: "
                f"{', '.join(missing_key)}\n"
                "  export LLM_ROUTER_API_KEY=<key>\n"
                "  (or skip those steps, or add --candidates-dry-run)"
            )

    print("stage-3 pipeline")
    print("  steps :", " -> ".join(selected))
    print("  python:", args.python)
    print()

    overall_start = time.time()

    for pos, step_name in enumerate(selected, 1):

        cmd = build_command(step_name, args, args.python)
        printable = " ".join(shlex.quote(c) for c in cmd)

        print(f"[{pos}/{len(selected)}] {step_name}")
        print(f"    $ {printable}")

        if args.dry_run:
            print()
            continue

        started = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - started

        if result.returncode != 0:
            print()
            print(f"!! step '{step_name}' failed (exit {result.returncode}) "
                  f"after {elapsed:.0f}s")
            print(f"   resume with:  python3 {Path(__file__).name} "
                  f"--from {step_name}")
            sys.exit(result.returncode)

        print(f"    done in {elapsed:.0f}s")
        print()

    if not args.dry_run:
        print(f"pipeline finished in {time.time() - overall_start:.0f}s")


if __name__ == "__main__":
    main()
