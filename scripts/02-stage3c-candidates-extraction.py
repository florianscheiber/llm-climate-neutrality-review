# scripts/02-stage3c-candidates-extraction.py  (stage 3c, step 2)
#
# SECOND-PASS extraction for the figures / tables that step 1
# (01-stage3c-data-extraction.py) flagged as "candidates for manual review"
# (see FIGURE AND TABLE CANDIDATES in prompts/stage3c_prompt.txt).
#
# WHY THIS EXISTS
#
# 01-stage3c-data-extraction.py works from the low-resolution figure REGIONS
# that pdf-to-markdown-extraction.py rasterises. Those regions are often
# cropped (axis / legend sliced off) or under-resolved, so the model
# reports the figure as a candidate instead of extracting it.
#
# This script goes back to the SOURCE PDF, locates each flagged figure /
# table by its caption, renders the FULL PAGE(S) it sits on at high DPI,
# and re-runs a focused extraction prompt (prompts/stage3c_candidates_prompt.txt)
# on that one figure -- passing the rows step 1 already got from it so
# the model does not duplicate them.
#
# INPUT
#
#     outputs/output-stage3c/<id>/<id>-candidates.txt      (flagged items)
#     outputs/output-stage3c/<id>/<id>-{mitigation,costs,emissions}.csv
#     inputs/fulltext_pdf/included-after-stage-3-100/<id>-main.pdf
#     inputs/fulltext_pdf/included-after-stage-3-100/<id>-supplement-1.pdf
#
# OUTPUT
#
#     outputs/output-stage3c-candidates/<id>/
#         <id>-mitigation.csv          same columns as stage3c
#         <id>-costs.csv
#         <id>-emissions.csv
#         <id>-plan.csv                one row per flagged item: located? skipped?
#         <id>_raw_response.txt        raw model JSON, one block per figure
#         pages/<pdf>-p0007.png        the full-page renders that were sent
#     outputs/output-stage3c-candidates/run_log.csv
#
# Then merge with the first pass:
#
#     python3 scripts/03-stage3c-merge-extractions.py \
#         outputs/output-stage3c outputs/output-stage3c \
#         --extra-input-dir outputs/output-stage3c-candidates
#
# SCOPE
#
# By default only "high-value extractable" candidates are attempted:
# structurally un-digitisable items (spatial maps with unlabelled
# cells/corridors, Sankey diagrams, violin/box plots, hopelessly
# overlapping series) are skipped -- a full-page render does not help
# those. Pure resolution / "axis not readable at this resolution"
# complaints ARE attempted, because that is exactly what re-rendering
# fixes. Use --all to attempt every candidate. Use --dry-run to just
# print the plan (location + pages) without calling the model.
#
# USAGE  (run from the repo root)
#
#     export LLM_ROUTER_API_KEY=<KEY>
#
#     # preview what would be done, no API calls:
#     python3 scripts/02-stage3c-candidates-extraction.py --dry-run
#
#     # run everything:
#     python3 scripts/02-stage3c-candidates-extraction.py
#
#     # a few papers only:
#     python3 scripts/02-stage3c-candidates-extraction.py --papers 287,436,315
#
# REQUIREMENTS
#
#     pip install pymupdf pillow truststore

from pathlib import Path
import argparse
import base64
import csv
import json
import os
import re
import time
import urllib.request
import urllib.error

import pymupdf
import truststore

truststore.inject_into_ssl()


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_MODEL = "openai/gpt-5.6-sol"
MAX_TOKENS = 30000
RETRIES = 3
RETRY_SECONDS = 5

# DPI for the full-page renders. 200 is what pdf-to-markdown uses for
# regions; we go higher because a whole page carries more detail.
PAGE_DPI = 300
# Downscale so the long edge never exceeds this (keeps the payload sane).
MAX_PAGE_DIM = 2400
MAX_IMAGE_PAYLOAD_MB = 28

DEFAULT_STAGE3C_DIR = Path("outputs/output-stage3c")
DEFAULT_OUTPUT_DIR = Path("outputs/output-stage3c-candidates")
DEFAULT_PDF_DIR = Path("inputs/fulltext_pdf/included-after-stage-3-100")
DEFAULT_PROMPT = (
    Path(__file__).resolve().parent.parent
    / "prompts" / "stage3c_candidates_prompt.txt"
)

# A candidate whose *reason* matches any of these is structurally
# un-digitisable: a sharper image will not help, so skip unless --all.
SKIP_REASON_PATTERNS = [
    r"\bsankey\b",
    r"\bviolin\b",
    r"\bbox[\-\s]?plot",
    r"\b3[\-\s]?d\b",
    r"three[\-\s]dimensional",
    r"dense scatter|scatter plot|point cloud",
    r"\bmap(s)?\b",
    r"network map|transmission map|spatial(ly)? (resolved|explicit)",
    r"grid cell|cell[\-\s]level|corridor[\-\s]level|corridor values",
    r"geographic identifier|unlabelled (grid )?cells",
    r"overlapping indistinguishable|hopelessly overlap",
    r"unlabelled axis|no readable axis|axis cannot be read",
]

# These columns MUST match stage3c-data-extraction.py exactly.
MITIGATION_COLUMNS = [
    "Mitigation_Measure", "Mitigation_Measure_Detail", "Scenario_Name",
    "Capacity", "Capacity_Unit", "Amount", "Amount_Unit",
    "Investment_Cost", "Investment_Cost_Unit",
    "Investment_Cost_Currency_Base_Year",
    "Variable_Cost", "Variable_Cost_Unit", "Variable_Cost_Currency_Base_Year",
    "Year", "Sector", "Sector_Other", "Area",
    "Source_Type", "Source_Number", "Comment",
]
COST_COLUMNS = [
    "Cost_Type", "Cost_Type_Detail", "Scenario_Name",
    "Value", "Value_Unit", "Currency_Base_Year",
    "Year", "Sector", "Sector_Other", "Area",
    "Source_Type", "Source_Number", "Comment",
]
EMISSION_COLUMNS = [
    "Emission_Type", "Emission_Type_Detail", "Scenario_Name",
    "Value", "Value_Unit",
    "Year", "Sector", "Sector_Other", "Area",
    "Source_Type", "Source_Number", "Comment",
]
COLUMNS = {
    "mitigation": MITIGATION_COLUMNS,
    "costs": COST_COLUMNS,
    "emissions": EMISSION_COLUMNS,
}


# ============================================================================
# CANDIDATE FILE PARSING
# ============================================================================

class Candidate:

    def __init__(self, raw_identifier, reason, info_type):
        self.raw_identifier = raw_identifier
        self.reason = reason
        self.info_type = info_type

    def __repr__(self):
        return f"<Candidate {self.raw_identifier!r}>"


def parse_candidates_file(path: Path):

    text = path.read_text(encoding="utf-8")
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]

    out = []

    for block in blocks:

        lines = block.splitlines()
        identifier = lines[0].strip()
        reason = ""
        info_type = ""

        for line in lines[1:]:
            low = line.lower()
            if low.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
            elif low.startswith("expected information type:"):
                info_type = line.split(":", 1)[1].strip()

        if identifier:
            out.append(Candidate(identifier, reason, info_type))

    return out


# A single figure/table target: kind ("figure"/"table"), a printable label
# ("4", "S7", "A.9", "B.12") and the search prefix / number.
class Target:

    def __init__(self, kind, prefix, number, parent: Candidate, supp=False):
        self.kind = kind
        self.prefix = prefix.upper()  # "", "S", "A", "B", "SI", ...
        self.number = number          # int
        self.parent = parent
        self.supp = supp              # identifier hinted "SI" / "supplementary"

    @property
    def label(self):
        if not self.prefix:
            return str(self.number)
        if self.prefix == "ED":
            return f"Extended Data {self.kind.capitalize()} {self.number}"
        if self.prefix in ("S", "SI", "SM"):
            return f"{self.prefix}{self.number}"
        return f"{self.prefix}.{self.number}"

    def search_prefixes(self):
        """Prefixes to try when locating the caption, best guess first."""
        if self.prefix == "ED":
            return ["ED"]
        if self.prefix in ("S", "SI", "SM"):
            # explicit supplementary label -- try the S-variants only, never
            # the bare number ("Figure S4" is not "Figure 4")
            return [self.prefix] + [
                p for p in ("S", "SI", "SM") if p != self.prefix
            ]
        if self.prefix:
            return [self.prefix]
        if self.supp:
            # bare number that the identifier tagged "SI" -- the real caption
            # may be "Figure 4" (in the SI) or "Figure S4"
            return ["S", "SI", "SM", ""]
        return [""]

    def __repr__(self):
        return f"<Target {self.kind} {self.label}>"


def expand_candidate_to_targets(cand: Candidate, max_range: int):
    """Turn one candidate line into concrete figure/table targets,
    expanding ranges like 'Figures B.12-B.15 and B.17-B.19'."""

    s = cand.raw_identifier.replace("–", "-").replace("—", "-")

    kind = "table" if re.search(r"\btab(?:le)?s?\b", s, re.I) else "figure"

    supp = bool(re.search(r"\b(SI|SM|supp(?:l(?:\.|ementary|ement)?)?)\b", s, re.I))
    extended = bool(re.search(r"extended\s+data", s, re.I))

    # Strip the kind words and connectives FIRST so "Figure"/"Fig." cannot
    # leak letters (e.g. the "s" of "Figures") into the reference parsing.
    s = re.sub(r"extended\s+data", " ", s, flags=re.I)
    s = re.sub(r"\bfig(?:ure)?s?\b\.?", " ", s, flags=re.I)
    s = re.sub(r"\btab(?:le)?s?\b\.?", " ", s, flags=re.I)
    s = re.sub(r"\b(and|panels?|SI|SM|supp\w*)\b", " ", s, flags=re.I)

    targets = []
    seen = set()

    def add(prefix, n):
        prefix = (prefix or "").upper()
        if len(prefix) > 2:
            prefix = ""
        if extended:
            prefix = "ED"
        key = (prefix, n)
        if key in seen:
            return
        seen.add(key)
        targets.append(Target(kind, prefix, n, cand, supp=supp))

    range_re = re.compile(
        r"([A-Za-z]{1,2})?\.?\s*(\d+)\s*-\s*([A-Za-z]{1,2})?\.?\s*(\d+)"
    )
    for m in range_re.finditer(s):
        p1, n1, p2, n2 = m.groups()
        prefix = (p1 or p2 or "")
        a, b = int(n1), int(n2)
        if 0 < b - a <= max_range:
            for n in range(a, b + 1):
                add(prefix, n)
    # blank out the ranges so their endpoints aren't re-read as standalones
    s = range_re.sub("  ", s)

    # standalone references ("4", "S4", "S7", "A.9", "B.12")
    for m in re.finditer(r"([A-Za-z]{1,2})?\.?\s*(\d+)", s):
        add(m.group(1) or "", int(m.group(2)))

    return targets


def is_low_value(cand: Candidate) -> bool:
    reason = cand.reason.lower()
    for pat in SKIP_REASON_PATTERNS:
        if re.search(pat, reason):
            return True
    return False


# ============================================================================
# EXISTING STAGE-3C ROWS FOR A FIGURE (so we don't duplicate them)
# ============================================================================

def load_existing_rows(stage3c_paper_dir: Path, folder_id: str):
    """Return {category: [row dict, ...]} for the first-pass CSVs."""
    out = {}
    for category, columns in COLUMNS.items():
        path = stage3c_paper_dir / f"{folder_id}-{category}.csv"
        rows = []
        if path.exists():
            with open(path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    rows.append(row)
        out[category] = rows
    return out


def rows_for_target(existing_rows, target: Target):
    """First-pass rows that already cite this figure/table."""
    wanted = target.label.lower().replace(".", "")
    hits = []
    for category, rows in existing_rows.items():
        for row in rows:
            st = (row.get("Source_Type") or "").strip().lower()
            sn = (row.get("Source_Number") or "").strip().lower().replace(".", "")
            if st == target.kind and sn == wanted:
                hits.append((category, row))
    return hits


# ============================================================================
# PDF: LOCATE FIGURE, RENDER PAGES
# ============================================================================

def build_caption_regex(target: Target, prefix=None):
    kind_word = r"tab(?:le)?s?" if target.kind == "table" else r"fig(?:ure|s)?"
    prefix = target.prefix if prefix is None else prefix
    if prefix == "ED":
        core = rf"extended\s+data\s+{kind_word}\.?\s*{target.number}"
    elif prefix:
        pre = re.escape(prefix)
        core = rf"{kind_word}\.?\s*{pre}[.\-\s]?\s*{target.number}"
    else:
        core = rf"{kind_word}\.?\s*{target.number}"
    # strict: the reference sits at the start of a line (optionally after a
    # markdown bold marker) and is followed by ". " + a caption word -- i.e.
    # it is the actual figure/table caption, not an in-text mention.
    strict = re.compile(
        rf"(?m)^[ \t>*_]*{core}\s*[.:|]\s+[\"'(A-Za-z0-9]",
        re.I,
    )
    # loose: reference followed by ". " / " | " + capital anywhere (may catch
    # a sentence that ends right after "Fig. 4.").
    caption = re.compile(
        rf"(?<![A-Za-z0-9]){core}\s*[.:|]\s+[\"'(A-Z]",
        re.I,
    )
    # any mention (last-resort fallback)
    mention = re.compile(rf"(?<![A-Za-z0-9]){core}(?![0-9])", re.I)
    return strict, caption, mention


_TOC_MARKER_RE = re.compile(
    r"list of (figures|tables|supplementary)|table of contents|^\s*contents\s*$",
    re.I | re.M,
)
_TOC_LINE_RE = re.compile(
    r"(?im)^\s*(supplementary\s+)?(figure|table|fig\.)\s*[A-Za-z]?\.?\s*\d+.*?"
    r"(\.\s*\.\s*\.|\s\d{1,3}\s*$)",
)


def is_toc_page(text: str) -> bool:
    """A contents / list-of-figures page: the caption strings live here but
    the figure itself does not."""
    if _TOC_MARKER_RE.search(text):
        return True
    return len(_TOC_LINE_RE.findall(text)) >= 5


def load_pdf_texts(pdf_paths):
    """Return {pdf_path: [(page_text, is_toc), ...]} -- read each PDF once."""
    cache = {}
    for pdf_path in pdf_paths:
        try:
            doc = pymupdf.open(pdf_path)
        except Exception:
            cache[pdf_path] = []
            continue
        pages = [page.get_text() or "" for page in doc]
        cache[pdf_path] = [(t, is_toc_page(t)) for t in pages]
        doc.close()
    return cache


def _locate_with_regexes(pdf_texts, strict_re, caption_re, mention_re):

    strict_hits, caption_hits, mention_hits = [], [], []

    for pdf_path, pages in pdf_texts.items():
        for i, (text, toc) in enumerate(pages):
            if not text or toc:
                continue
            if strict_re.search(text):
                strict_hits.append((pdf_path, i + 1))
            elif caption_re.search(text):
                caption_hits.append((pdf_path, i + 1))
            elif mention_re.search(text):
                mention_hits.append((pdf_path, i + 1))

    for hits, cap, quality in (
        (strict_hits, 2, "caption"),
        (caption_hits, 3, "caption-loose"),
        (mention_hits, 1, "mention"),
    ):
        if hits:
            by_pdf = {}
            for pdf_path, pageno in hits:
                by_pdf.setdefault(pdf_path, []).append(pageno)
            pdf_path = max(by_pdf, key=lambda p: len(by_pdf[p]))
            return pdf_path, sorted(set(by_pdf[pdf_path]))[:cap], quality

    return None, [], ""


def locate_target_pages(pdf_texts, target: Target):
    """Return (pdf_path, [page_number, ...], match_quality) for the pages to
    render, or (None, [], "") if the figure could not be located.

    Tries the target's own prefix first, then plausible supplementary
    prefixes ("S", "SI", ...) when the identifier hinted at that. Stops at
    the first prefix that yields a caption-quality hit."""

    fallback = (None, [], "")

    for prefix in target.search_prefixes():
        strict_re, caption_re, mention_re = build_caption_regex(target, prefix)
        pdf_path, pages, quality = _locate_with_regexes(
            pdf_texts, strict_re, caption_re, mention_re
        )
        if quality in ("caption", "caption-loose"):
            return pdf_path, pages, quality
        if pdf_path and fallback[0] is None:
            fallback = (pdf_path, pages, quality)

    return fallback


def render_pages(pdf_path: Path, page_numbers, out_dir: Path):
    """Render the given 1-based page numbers to PNG, return the file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    doc = pymupdf.open(pdf_path)
    try:
        zoom = PAGE_DPI / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        for pageno in page_numbers:
            if pageno < 1 or pageno > len(doc):
                continue
            page = doc[pageno - 1]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            if max(pix.width, pix.height) > MAX_PAGE_DIM:
                scale = MAX_PAGE_DIM / max(pix.width, pix.height)
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(zoom * scale, zoom * scale),
                    alpha=False,
                )
            name = f"{pdf_path.stem}-p{pageno:04d}.png"
            path = out_dir / name
            pix.save(path)
            written.append(path)
    finally:
        doc.close()
    return written


# ============================================================================
# CAPTION TEXT (for the prompt)
# ============================================================================

def extract_caption_text(pdf_texts, pdf_path: Path, page_numbers, target: Target):
    pages = pdf_texts.get(pdf_path, [])
    for prefix in target.search_prefixes():
        strict_re, caption_re, _ = build_caption_regex(target, prefix)
        for pageno in page_numbers:
            if pageno < 1 or pageno > len(pages):
                continue
            text = pages[pageno - 1][0]
            m = strict_re.search(text) or caption_re.search(text)
            if not m:
                continue
            snippet = text[m.start():m.start() + 600].split("\n\n")[0]
            return " ".join(snippet.split())
    return ""


# ============================================================================
# OPENROUTER
# ============================================================================

def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_openrouter_vision(prompt_text, image_paths, model, api_key):

    content = [{"type": "text", "text": prompt_text}]
    for image_path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_to_base64(image_path)}"
            },
        })

    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": content}],
    }
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url="https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "stage3c-candidates-extraction",
            "X-Title": "stage3c-candidates-extraction",
        },
        method="POST",
    )

    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
                if "choices" not in body:
                    print(json.dumps(body, indent=2))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTPError {exc.code}: {detail}")
            if attempt < RETRIES:
                time.sleep(RETRY_SECONDS)
                continue
            raise last_error
        except Exception as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(RETRY_SECONDS)
                continue
            raise

    raise last_error


def parse_llm_output(raw_text):
    raw_text = raw_text.strip()
    if not raw_text.startswith("{"):
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1:
            raw_text = raw_text[start:end + 1]
    return json.loads(raw_text)


# ============================================================================
# PROMPT ASSEMBLY
# ============================================================================

def build_prompt(base_prompt, target: Target, existing_hits):

    lines = [base_prompt, "", "=" * 72, ""]

    lines.append(f"TARGET: {target.kind.capitalize()} {target.label}")
    lines.append(f"Source_Number to use for every row: {target.label}")
    lines.append(f"Source_Type to use for every row: {target.kind.capitalize()}")
    lines.append("")
    lines.append(f"Flagged because: {target.parent.reason}")
    lines.append(f"Expected information type: {target.parent.info_type}")
    lines.append("")

    if existing_hits:
        lines.append(
            "ROWS THE FIRST PASS ALREADY EXTRACTED FROM THIS "
            f"{target.kind.upper()} -- do NOT repeat these, only add what is "
            "missing:"
        )
        for category, row in existing_hits:
            summary = {
                k: v for k, v in row.items()
                if v not in ("", None) and k not in (
                    "paper_id", "Title", "Publication_Year", "DOI"
                )
            }
            lines.append(f"  [{category}] {json.dumps(summary, ensure_ascii=False)}")
    else:
        lines.append(
            "The first pass extracted NOTHING from this "
            f"{target.kind}; everything readable here is new."
        )

    lines.append("")
    lines.append(
        "The attached image(s) are full-page renders. Find "
        f"{target.kind} {target.label} on the page and extract only it."
    )

    return "\n".join(lines)


# ============================================================================
# CSV
# ============================================================================

def write_csv(path: Path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def normalize_row(row):
    if not isinstance(row, dict):
        return row
    out = dict(row)
    if not out.get("Area") and out.get("Region"):
        out["Area"] = out["Region"]
    if not out.get("Source_Number") and out.get("Source_Detail"):
        out["Source_Number"] = out["Source_Detail"]
    return out


# ============================================================================
# RUN LOG
# ============================================================================

RUN_LOG_COLUMNS = [
    "paper_id", "identifier", "target", "status", "message",
    "pdf", "pages", "match_quality", "images_sent",
    "mitigation_rows", "cost_rows", "emission_rows",
    "existing_rows_for_target", "runtime_seconds",
]


def append_run_log(path: Path, record):
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS, delimiter=";")
        if not exists:
            writer.writeheader()
        writer.writerow({c: record.get(c, "") for c in RUN_LOG_COLUMNS})


# ============================================================================
# PER-PAPER PROCESSING
# ============================================================================

def process_paper(
    folder_id,
    stage3c_dir: Path,
    pdf_dir: Path,
    output_dir: Path,
    base_prompt: str,
    model: str,
    api_key: str,
    attempt_all: bool,
    dry_run: bool,
    max_range: int,
    run_log_path: Path,
    force: bool = False,
):
    paper_id = str(int(folder_id))
    stage3c_paper_dir = stage3c_dir / folder_id
    candidates_file = stage3c_paper_dir / f"{folder_id}-candidates.txt"

    if not candidates_file.exists() or candidates_file.stat().st_size == 0:
        return

    candidates = parse_candidates_file(candidates_file)
    if not candidates:
        return

    paper_out_dir = output_dir / folder_id
    if (
        not dry_run
        and not force
        and (paper_out_dir / f"{folder_id}-plan.csv").exists()
        and (paper_out_dir / f"{folder_id}-mitigation.csv").exists()
    ):
        print(f"{folder_id}: already processed (use --force to redo)")
        return

    pdf_paths = sorted(pdf_dir.glob(f"{folder_id}-*.pdf"))
    if not pdf_paths:
        print(f"{folder_id}: no source PDF found, skipping")
        append_run_log(run_log_path, {
            "paper_id": paper_id, "identifier": "", "target": "",
            "status": "error", "message": "no source pdf",
        })
        return

    existing_rows = load_existing_rows(stage3c_paper_dir, folder_id)
    pdf_texts = load_pdf_texts(pdf_paths)

    pages_dir = paper_out_dir / "pages"

    plan_rows = []
    collected = {"mitigation": [], "costs": [], "emissions": []}
    raw_blocks = []
    seen_row_keys = {"mitigation": set(), "costs": set(), "emissions": set()}

    # de-dupe targets across candidate lines
    done_targets = set()

    for cand in candidates:

        low_value = is_low_value(cand)
        targets = expand_candidate_to_targets(cand, max_range)

        if not targets:
            plan_rows.append({
                "identifier": cand.raw_identifier, "target": "",
                "info_type": cand.info_type, "low_value_skip": False,
                "existing_rows_for_target": 0, "pdf": "", "pages": "",
                "match_quality": "",
                "status": "unparsed_identifier",
                "message": "could not derive a figure/table number "
                           "(range too large?)",
            })
            append_run_log(run_log_path, {
                "paper_id": paper_id, "identifier": cand.raw_identifier,
                "target": "", "status": "unparsed_identifier",
                "message": cand.raw_identifier[:120],
            })
            continue

        for target in targets:

            tkey = (target.kind, target.label.lower())
            if tkey in done_targets:
                continue
            done_targets.add(tkey)

            existing_hits = rows_for_target(existing_rows, target)

            plan = {
                "identifier": cand.raw_identifier,
                "target": f"{target.kind} {target.label}",
                "info_type": cand.info_type,
                "low_value_skip": low_value and not attempt_all,
                "existing_rows_for_target": len(existing_hits),
                "pdf": "",
                "pages": "",
                "match_quality": "",
                "status": "",
                "message": "",
            }

            if low_value and not attempt_all:
                plan["status"] = "skipped_low_value"
                plan["message"] = cand.reason[:120]
                plan_rows.append(plan)
                append_run_log(run_log_path, {
                    "paper_id": paper_id,
                    "identifier": cand.raw_identifier,
                    "target": plan["target"],
                    "status": "skipped_low_value",
                    "message": cand.reason[:200],
                    "existing_rows_for_target": len(existing_hits),
                })
                continue

            pdf_path, pages, match_quality = locate_target_pages(
                pdf_texts, target
            )
            plan["match_quality"] = match_quality

            if not pdf_path:
                plan["status"] = "not_located"
                plan["message"] = "caption not found in main or supplement pdf"
                plan_rows.append(plan)
                append_run_log(run_log_path, {
                    "paper_id": paper_id,
                    "identifier": cand.raw_identifier,
                    "target": plan["target"],
                    "status": "not_located",
                    "message": "caption not found",
                    "existing_rows_for_target": len(existing_hits),
                })
                continue

            plan["pdf"] = pdf_path.name
            plan["pages"] = ",".join(str(p) for p in pages)

            if dry_run:
                plan["status"] = "would_extract"
                plan_rows.append(plan)
                append_run_log(run_log_path, {
                    "paper_id": paper_id,
                    "identifier": cand.raw_identifier,
                    "target": plan["target"],
                    "status": "would_extract",
                    "message": "",
                    "pdf": pdf_path.name,
                    "pages": plan["pages"],
                    "match_quality": match_quality,
                    "existing_rows_for_target": len(existing_hits),
                })
                continue

            start_time = time.time()

            image_paths = render_pages(pdf_path, pages, pages_dir)
            if not image_paths:
                plan["status"] = "render_failed"
                plan_rows.append(plan)
                append_run_log(run_log_path, {
                    "paper_id": paper_id,
                    "identifier": cand.raw_identifier,
                    "target": plan["target"],
                    "status": "render_failed",
                    "message": "",
                    "pdf": pdf_path.name, "pages": plan["pages"],
                })
                continue

            total_mb = sum(p.stat().st_size for p in image_paths) / 1024 / 1024
            if total_mb > MAX_IMAGE_PAYLOAD_MB and len(image_paths) > 1:
                image_paths = image_paths[:1]

            caption_text = extract_caption_text(pdf_texts, pdf_path, pages, target)

            prompt_text = build_prompt(base_prompt, target, existing_hits)
            if caption_text:
                prompt_text += f"\n\nCaption as printed:\n{caption_text}"

            try:
                raw = call_openrouter_vision(
                    prompt_text, image_paths, model, api_key
                )
            except Exception as exc:
                plan["status"] = "api_error"
                plan["message"] = str(exc)[:160]
                plan_rows.append(plan)
                append_run_log(run_log_path, {
                    "paper_id": paper_id,
                    "identifier": cand.raw_identifier,
                    "target": plan["target"],
                    "status": "api_error",
                    "message": str(exc)[:200],
                    "pdf": pdf_path.name, "pages": plan["pages"],
                    "runtime_seconds": round(time.time() - start_time, 1),
                })
                continue

            raw_blocks.append(
                f"==================== {target.kind.upper()} {target.label} "
                f"({pdf_path.name} p{plan['pages']}) "
                f"====================\n{raw}\n"
            )

            try:
                result = parse_llm_output(raw)
            except Exception as exc:
                plan["status"] = "unparseable_json"
                plan["message"] = str(exc)[:160]
                plan_rows.append(plan)
                append_run_log(run_log_path, {
                    "paper_id": paper_id,
                    "identifier": cand.raw_identifier,
                    "target": plan["target"],
                    "status": "unparseable_json",
                    "message": str(exc)[:200],
                    "pdf": pdf_path.name, "pages": plan["pages"],
                    "runtime_seconds": round(time.time() - start_time, 1),
                })
                continue

            n_new = {"mitigation": 0, "costs": 0, "emissions": 0}

            if result.get("eligible", True):
                for category in ("mitigation", "costs", "emissions"):
                    for row in result.get(category, []) or []:
                        row = normalize_row(row)
                        if not isinstance(row, dict):
                            continue
                        # force the target's source identity
                        row["Source_Type"] = target.kind.capitalize()
                        if not row.get("Source_Number"):
                            row["Source_Number"] = target.label
                        key = json.dumps(row, sort_keys=True, default=str)
                        if key in seen_row_keys[category]:
                            continue
                        seen_row_keys[category].add(key)
                        collected[category].append(row)
                        n_new[category] += 1
                status = "extracted" if sum(n_new.values()) else "no_new_rows"
                message = result.get("reason", "")[:160]
            else:
                status = "model_declined"
                message = result.get("reason", "")[:200]

            plan["status"] = status
            plan["message"] = message
            plan_rows.append(plan)

            append_run_log(run_log_path, {
                "paper_id": paper_id,
                "identifier": cand.raw_identifier,
                "target": plan["target"],
                "status": status,
                "message": message,
                "pdf": pdf_path.name,
                "pages": plan["pages"],
                "match_quality": match_quality,
                "images_sent": len(image_paths),
                "mitigation_rows": n_new["mitigation"],
                "cost_rows": n_new["costs"],
                "emission_rows": n_new["emissions"],
                "existing_rows_for_target": len(existing_hits),
                "runtime_seconds": round(time.time() - start_time, 1),
            })

            print(
                f"{folder_id}: {target.kind} {target.label} -> {status} "
                f"(+{n_new['mitigation']}m/+{n_new['costs']}c/+{n_new['emissions']}e)"
            )

    if not plan_rows:
        return

    paper_out_dir.mkdir(parents=True, exist_ok=True)

    with open(paper_out_dir / f"{folder_id}-plan.csv", "w",
              encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "identifier", "target", "info_type", "low_value_skip",
                "existing_rows_for_target", "pdf", "pages", "match_quality",
                "status", "message",
            ],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(plan_rows)

    if dry_run:
        return

    for category, columns in COLUMNS.items():
        write_csv(
            paper_out_dir / f"{folder_id}-{category}.csv",
            columns,
            collected[category],
        )

    if raw_blocks:
        (paper_out_dir / f"{folder_id}_raw_response.txt").write_text(
            "\n".join(raw_blocks), encoding="utf-8"
        )

    print(
        f"{folder_id}: wrote "
        f"{len(collected['mitigation'])}m / {len(collected['costs'])}c / "
        f"{len(collected['emissions'])}e new rows"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Second-pass extraction of stage3c candidate figures/tables"
    )
    parser.add_argument("--stage3c-dir", default=str(DEFAULT_STAGE3C_DIR))
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--papers",
        default=None,
        help="comma-separated folder ids (e.g. 287,436). default: all",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="also attempt candidates flagged as structurally un-digitisable",
    )
    parser.add_argument(
        "--max-range",
        type=int,
        default=6,
        help="max figures to expand from a range like 'Figures S1-S40' "
             "(default 6; larger ranges are skipped unless --all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="locate figures and write the per-paper plan.csv, no API calls",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-process papers that already have output (default: skip them)",
    )
    args = parser.parse_args()

    stage3c_dir = Path(args.stage3c_dir)
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)

    api_key = os.environ.get("LLM_ROUTER_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("LLM_ROUTER_API_KEY not set (use --dry-run to preview)")

    base_prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_dir / (
        "run_log_dry.csv" if args.dry_run else "run_log.csv"
    )
    # the dry-run log is a fresh snapshot every time; the live log is
    # appended so an interrupted run can resume.
    if args.dry_run and run_log_path.exists():
        run_log_path.unlink()

    if args.papers:
        wanted = {p.strip().zfill(3) for p in args.papers.split(",")}
        folder_ids = sorted(wanted)
    else:
        folder_ids = [
            d.name for d in sorted(stage3c_dir.iterdir())
            if d.is_dir() and re.fullmatch(r"\d+", d.name)
        ]

    max_range = args.max_range if not args.all else max(args.max_range, 60)

    print(
        f"Papers to scan: {len(folder_ids)}  "
        f"(mode: {'DRY RUN' if args.dry_run else 'live'}, "
        f"scope: {'ALL candidates' if args.all else 'high-value only'})"
    )

    for folder_id in folder_ids:
        try:
            process_paper(
                folder_id=folder_id,
                stage3c_dir=stage3c_dir,
                pdf_dir=pdf_dir,
                output_dir=output_dir,
                base_prompt=base_prompt,
                model=args.model,
                api_key=api_key or "",
                attempt_all=args.all,
                dry_run=args.dry_run,
                force=args.force,
                max_range=max_range,
                run_log_path=run_log_path,
            )
        except Exception as exc:
            print(f"ERROR {folder_id}: {exc}")
            append_run_log(run_log_path, {
                "paper_id": str(int(folder_id)),
                "identifier": "", "target": "",
                "status": "error", "message": str(exc)[:200],
            })

    print(f"\nRun log: {run_log_path}")


if __name__ == "__main__":
    main()
