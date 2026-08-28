# scripts/01-stage3c-data-extraction.py  (stage 3c, step 1)
#
# INPUT:
#
#     Markdown produced by pdf-to-markdown-extraction.py, e.g.
#
#     intermediate/pdf-extraction/
#     └── 138/
#         ├── 138-main.md
#         ├── 138-supplement-1.md
#         └── images/
#             ├── 138-main.pdf-0002-01.png
#             └── ...
#
# OUTPUT:
#
#     output/
#     └── 138/
#         ├── 138-mitigation.csv
#         ├── 138-costs.csv
#         ├── 138-emissions.csv
#         ├── 138-candidates.txt
#         ├── 138_document_content.md
#         └── 138_raw_response.txt
#
# APPROACH:
#
# Stage 03c works from the Markdown conversion of each paper, NOT the PDF.
#
# The Markdown already contains:
#
# - running text in reading order
# - tables reconstructed as Markdown tables
# - figure / table captions
# - image links to the figures extracted from the PDF
#
# The Markdown is sent together with the extracted figure images, so the
# model can also read the figures directly (vision). The figures are sent
# in several calls of at most IMAGE_BATCH_SIZE images each (the Markdown is
# repeated on every call); the per-call results are merged and exact
# duplicate rows are dropped.
#
# REQUIREMENTS:
#
# pip install pillow
# pip install truststore
#
# PROMPT:
#
# prompts/stage3c_prompt.txt
#
# USAGE IN TERMINAL:  (run from the repo root)
#
# export LLM_ROUTER_API_KEY=<KEY>
#
# python3 scripts/01-stage3c-data-extraction.py \
# markdown-directory \
# output-directory
#
# EXAMPLE:
#
# export LLM_ROUTER_API_KEY=<KEY>
#
# python3 scripts/01-stage3c-data-extraction.py \
# intermediate/pdf-extraction \
# outputs/output-stage3c
#
# Use --limit N to process only the first N papers (article-id order).
#
# NOTES:
#
# - All Markdown files belonging to the same article ID are processed together.
# - Main paper and supplementary Markdown are merged into a single document context.
# - Figure images are downscaled and sent in batches of at most
#   IMAGE_BATCH_SIZE; a batch that still exceeds MAX_IMAGE_PAYLOAD_MB is
#   downscaled further.
# - Already processed papers are skipped automatically if all three
#   output CSVs already exist.
#

from pathlib import Path
import argparse
import base64
import csv
import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
import urllib.error

from PIL import Image

import truststore

truststore.inject_into_ssl()


# ============================================================================
# CONFIG
# ============================================================================

#DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_MODEL = "openai/gpt-5.6-sol"
MODE = "iterative_images"
MAX_TOKENS = 30000
RETRIES = 3
RETRY_SECONDS = 5

PRIMARY_MAX_DIM = 2000
FALLBACK_MAX_DIM = 1800
MAX_IMAGE_PAYLOAD_MB = 30

# Figures are sent in several vision calls of at most this many images each.
# The Markdown is repeated on every call; the per-batch results are merged
# and exact-duplicate rows are dropped.
IMAGE_BATCH_SIZE = 20


# ============================================================================
# HELPERS
# ============================================================================

def debug(msg):

    if os.environ.get("LLM_DEBUG") == "1":
        print(f"[DEBUG] {msg}")


def load_prompt(prompt_path: Path) -> str:

    with open(
        prompt_path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


def extract_article_id(name: str) -> str:

    match = re.match(
        r"^(\d+)",
        name
    )

    if not match:
        raise RuntimeError(
            f"Could not extract article id from {name}"
        )

    return match.group(1)


def group_markdown_by_article_id(markdown_folder: Path):

    groups = {}

    for entry in sorted(markdown_folder.iterdir()):

        if not entry.is_dir():
            continue

        if not re.fullmatch(r"\d+", entry.name):
            continue

        markdown_files = sorted(
            entry.glob("*.md")
        )

        if not markdown_files:
            continue

        image_files = sorted(
            (entry / "images").glob("*.png")
        )

        groups[entry.name] = {
            "markdown": markdown_files,
            "images": image_files,
        }

    return groups


def build_document_content(markdown_files):

    parts = []

    for markdown_path in markdown_files:

        parts.append(
            f"\n\n====================\n"
            f"DOCUMENT: {markdown_path.name}\n"
            f"====================\n"
        )

        parts.append(
            markdown_path.read_text(
                encoding="utf-8"
            )
        )

    return "\n".join(parts)


def count_pages(document_content: str) -> int:

    return len(
        re.findall(
            r"^## Page \d+",
            document_content,
            re.MULTILINE,
        )
    )


# ============================================================================
# IMAGE PREPARATION
# ============================================================================

def prepare_images(
    image_paths,
    target_dir: Path,
    max_dim: int,
):

    prepared = []

    for image_path in image_paths:

        img = Image.open(image_path)

        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        largest = max(img.size)

        if largest > max_dim:

            scale = max_dim / largest

            img = img.resize(
                (
                    int(img.size[0] * scale),
                    int(img.size[1] * scale),
                ),
                Image.LANCZOS,
            )

        output_file = target_dir / image_path.name

        img.save(output_file)

        prepared.append(output_file)

    return prepared


def get_total_image_size_mb(image_paths):

    total_bytes = 0

    for image_path in image_paths:

        total_bytes += image_path.stat().st_size

    return total_bytes / 1024 / 1024


def chunked(sequence, size):

    for start in range(0, len(sequence), size):
        yield sequence[start:start + size]


# ============================================================================
# OPENROUTER
# ============================================================================

def image_to_base64(image_path: Path) -> str:

    with open(image_path, "rb") as f:

        return base64.b64encode(f.read()).decode("utf-8")


def call_openrouter_vision(
    document_content,
    image_paths,
    prompt,
    model,
    api_key,
):

    content = [
        {
            "type": "text",
            "text": prompt + "\n\n" + document_content,
        }
    ]

    for image_path in image_paths:

        image_b64 = image_to_base64(image_path)

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                },
            }
        )

    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url="https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "stage3c-extraction",
            "X-Title": "stage3c-extraction",
        },
        method="POST",
    )

    last_error = None

    for attempt in range(1, RETRIES + 1):

        try:

            with urllib.request.urlopen(
                request,
                timeout=300
            ) as response:

                payload = json.loads(
                    response.read().decode("utf-8")
                )

                if "choices" not in payload:
                    print(json.dumps(payload, indent=2))

                return (
                    payload["choices"][0]
                    ["message"]
                    ["content"]
                )

        except urllib.error.HTTPError as exc:

            body = exc.read().decode(
                "utf-8",
                errors="replace"
            )

            last_error = RuntimeError(
                f"HTTPError {exc.code}: {body}"
            )

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


# ============================================================================
# JSON
# ============================================================================

def parse_llm_output(raw_text):

    raw_text = raw_text.strip()

    if not raw_text.startswith("{"):

        start = raw_text.find("{")
        end = raw_text.rfind("}")

        if start != -1 and end != -1:
            raw_text = raw_text[start:end + 1]

    return json.loads(raw_text)


def normalize_output_rows(result):

    for section_name in [
        "mitigation",
        "costs",
        "emissions",
    ]:

        rows = result.get(section_name, [])

        if not isinstance(rows, list):
            continue

        normalized_rows = []

        for row in rows:

            if not isinstance(row, dict):
                normalized_rows.append(row)
                continue

            normalized_row = dict(row)

            if (
                not normalized_row.get("Area")
                and normalized_row.get("Region")
            ):
                normalized_row["Area"] = normalized_row["Region"]

            if (
                not normalized_row.get("Source_Number")
                and normalized_row.get("Source_Detail")
            ):
                normalized_row["Source_Number"] = normalized_row["Source_Detail"]

            normalized_rows.append(normalized_row)

        result[section_name] = normalized_rows

    return result


def _row_key(row):

    if isinstance(row, dict):
        return json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

    return str(row)


def merge_batch_results(batch_results):
    """Merge the per-batch JSON payloads into one result.

    The Markdown is sent on every batch, so the text / table datapoints are
    re-extracted each time; exact-duplicate rows and repeated figure/table
    candidates are dropped here.
    """

    merged = {
        "mitigation": [],
        "costs": [],
        "emissions": [],
        "candidates": [],
    }

    seen_rows = {
        "mitigation": set(),
        "costs": set(),
        "emissions": set(),
    }

    seen_candidates = set()
    eligible_votes = []

    for result in batch_results:

        if not isinstance(result, dict):
            continue

        eligible_votes.append(
            result.get("eligible", True)
        )

        for section in ("mitigation", "costs", "emissions"):

            rows = result.get(section, [])

            if not isinstance(rows, list):
                continue

            for row in rows:

                row_key = _row_key(row)

                if row_key in seen_rows[section]:
                    continue

                seen_rows[section].add(row_key)
                merged[section].append(row)

        candidates = result.get("candidates", [])

        if not isinstance(candidates, list):
            continue

        for candidate in candidates:

            identifier = ""

            if isinstance(candidate, dict):
                identifier = str(
                    candidate.get("identifier", "")
                ).strip().lower()

            if identifier and identifier in seen_candidates:
                continue

            if identifier:
                seen_candidates.add(identifier)

            merged["candidates"].append(candidate)

    merged["eligible"] = (
        any(eligible_votes)
        if eligible_votes
        else True
    )

    return merged


# ============================================================================
# CSV WRITING
# ============================================================================

# These MUST stay in sync with the OUTPUT FORMAT block in prompts/stage3c_prompt.txt.

MITIGATION_COLUMNS = [
    "Mitigation_Measure",
    "Mitigation_Measure_Detail",
    "Scenario_Name",
    "Capacity",
    "Capacity_Unit",
    "Amount",
    "Amount_Unit",
    "Investment_Cost",
    "Investment_Cost_Unit",
    "Investment_Cost_Currency_Base_Year",
    "Variable_Cost",
    "Variable_Cost_Unit",
    "Variable_Cost_Currency_Base_Year",
    "Year",
    "Sector",
    "Sector_Other",
    "Area",
    "Source_Type",
    "Source_Number",
    "Comment",
]

COST_COLUMNS = [
    "Cost_Type",
    "Cost_Type_Detail",
    "Scenario_Name",
    "Value",
    "Value_Unit",
    "Currency_Base_Year",
    "Year",
    "Sector",
    "Sector_Other",
    "Area",
    "Source_Type",
    "Source_Number",
    "Comment",
]

EMISSION_COLUMNS = [
    "Emission_Type",
    "Emission_Type_Detail",
    "Scenario_Name",
    "Value",
    "Value_Unit",
    "Year",
    "Sector",
    "Sector_Other",
    "Area",
    "Source_Type",
    "Source_Number",
    "Comment",
]


def write_csv(path, columns, rows):

    with open(
        path,
        "w",
        encoding="utf-8",
        newline=""
    ) as fout:

        writer = csv.DictWriter(
            fout,
            fieldnames=columns,
            delimiter=";"
        )

        writer.writeheader()

        for row in rows:

            clean_row = {}

            for col in columns:
                clean_row[col] = row.get(col, "")

            writer.writerow(clean_row)


# ============================================================================
# PAPER PROCESSING
# ============================================================================

def append_log(
    log_file,
    paper_id,
    eligible,
    status,
    message,
    mitigation_rows="",
    cost_rows="",
    emission_rows="",
    runtime_seconds="",
    document_mode="",
    markdown_count="",
    page_count="",
    image_count="",
    image_batches="",
):

    file_exists = log_file.exists()

    with open(
        log_file,
        "a",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(
            f,
            delimiter=";"
        )

        if not file_exists:
            writer.writerow([
                "paper_id",
                "eligible",
                "status",
                "message",
                "mitigation_rows",
                "cost_rows",
                "emission_rows",
                "runtime_seconds",
                "document_mode",
                "markdown_count",
                "page_count",
                "image_count",
                "image_batches",
            ])

        writer.writerow([
            paper_id,
            eligible,
            status,
            message,
            mitigation_rows,
            cost_rows,
            emission_rows,
            runtime_seconds,
            document_mode,
            markdown_count,
            page_count,
            image_count,
            image_batches,
        ])


def process_paper(
    article_id,
    markdown_files,
    image_files,
    output_root: Path,
    prompt_text: str,
    model: str,
    api_key: str,
):
    start_time = time.time()

    debug(f"Processing {article_id}")

    article_output_dir = output_root / article_id

    mitigation_csv = article_output_dir / f"{article_id}-mitigation.csv"
    costs_csv = article_output_dir / f"{article_id}-costs.csv"
    emissions_csv = article_output_dir / f"{article_id}-emissions.csv"

    if (
        mitigation_csv.exists()
        and costs_csv.exists()
        and emissions_csv.exists()
    ):
        print(f"{article_id}: already processed")
        return None

    article_output_dir.mkdir(parents=True, exist_ok=True)

    main_markdown = [
        p for p in markdown_files
        if p.stem.lower().endswith("-main")
    ]

    if len(main_markdown) != 1:
        raise RuntimeError(
            "no unique main markdown found"
        )

    supplement_markdown = [
        p for p in markdown_files
        if p not in main_markdown
    ]

    selected_markdown = main_markdown + supplement_markdown

    document_mode = "markdown_plus_images"
    markdown_count = len(selected_markdown)

    document_content = build_document_content(selected_markdown)
    page_count = count_pages(document_content)

    with open(
        article_output_dir / f"{article_id}_document_content.md",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(document_content)

    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"{article_id}_")
    )

    # The raw model response is written incrementally, one block per batch,
    # BEFORE any JSON parsing -- so it survives a parse error or a crash
    # part way through the batches.
    raw_response_path = (
        article_output_dir / f"{article_id}_raw_response.txt"
    )
    raw_file = open(raw_response_path, "w", encoding="utf-8")

    batch_results = []

    try:

        work_dir = temp_dir / "images"
        work_dir.mkdir()

        prepared_images = prepare_images(
            image_files,
            work_dir,
            PRIMARY_MAX_DIM,
        )

        if not prepared_images:
            document_mode = "markdown_only"

        batches = list(
            chunked(prepared_images, IMAGE_BATCH_SIZE)
        ) or [[]]

        debug(
            f"{article_id}: {len(prepared_images)} images "
            f"in {len(batches)} batch(es) of <= {IMAGE_BATCH_SIZE}"
        )

        for batch_index, batch in enumerate(batches, start=1):

            if get_total_image_size_mb(batch) > MAX_IMAGE_PAYLOAD_MB:

                fallback_dir = temp_dir / f"batch_{batch_index:02d}_fb"
                fallback_dir.mkdir()

                batch = prepare_images(
                    batch,
                    fallback_dir,
                    FALLBACK_MAX_DIM,
                )

            debug(
                f"{article_id}: batch {batch_index}/{len(batches)} "
                f"({len(batch)} images)"
            )

            raw_response = call_openrouter_vision(
                document_content=document_content,
                image_paths=batch,
                prompt=prompt_text,
                model=model,
                api_key=api_key,
            )

            raw_file.write(
                f"==================== BATCH "
                f"{batch_index}/{len(batches)} "
                f"({len(batch)} images) ====================\n"
                f"{raw_response}\n\n"
            )
            raw_file.flush()

            try:
                batch_results.append(
                    parse_llm_output(raw_response)
                )
            except Exception as exc:
                print(
                    f"{article_id}: batch {batch_index} "
                    f"returned unparseable JSON ({exc})"
                )
                raw_file.write(
                    f"[batch {batch_index}: JSON parse failed: {exc}]\n\n"
                )
                raw_file.flush()

    finally:

        raw_file.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not batch_results:
        raise RuntimeError(
            "no batch returned parseable JSON "
            f"(raw response saved to {raw_response_path.name})"
        )

    image_batches = len(batches)

    result = merge_batch_results(batch_results)
    result = normalize_output_rows(result)

    candidate_file = article_output_dir / f"{article_id}-candidates.txt"

    with open(
        candidate_file,
        "w",
        encoding="utf-8"
    ) as f:

        for candidate in result.get("candidates", []):
            f.write(
                f"{candidate.get('identifier', '')}\n"
                f"Reason: {candidate.get('reason', '')}\n"
                f"Expected information type: "
                f"{candidate.get('expected_information_type', '')}\n\n"
            )

    eligible = result.get("eligible", True)

    mitigation_rows = result.get("mitigation", [])
    cost_rows = result.get("costs", [])
    emission_rows = result.get("emissions", [])

    if not eligible:

        write_csv(mitigation_csv, MITIGATION_COLUMNS, [])
        write_csv(costs_csv, COST_COLUMNS, [])
        write_csv(emissions_csv, EMISSION_COLUMNS, [])

        print(f"{article_id}: not eligible")

        runtime_seconds = round(time.time() - start_time, 1)

        return {
            "paper_id": article_id,
            "eligible": False,
            "status": "saved",
            "message": "no climate-neutral scenario found",
            "mitigation_rows": 0,
            "cost_rows": 0,
            "emission_rows": 0,
            "runtime_seconds": runtime_seconds,
            "document_mode": document_mode,
            "markdown_count": markdown_count,
            "page_count": page_count,
            "image_count": len(prepared_images),
            "image_batches": image_batches,
        }

    write_csv(mitigation_csv, MITIGATION_COLUMNS, mitigation_rows)
    write_csv(costs_csv, COST_COLUMNS, cost_rows)
    write_csv(emissions_csv, EMISSION_COLUMNS, emission_rows)

    print(f"{article_id}: completed")

    runtime_seconds = round(time.time() - start_time, 1)

    return {
        "paper_id": article_id,
        "eligible": True,
        "status": "saved",
        "message": "ok",
        "mitigation_rows": len(mitigation_rows),
        "cost_rows": len(cost_rows),
        "emission_rows": len(emission_rows),
        "runtime_seconds": runtime_seconds,
        "document_mode": document_mode,
        "markdown_count": markdown_count,
        "page_count": page_count,
        "image_count": len(prepared_images),
        "image_batches": image_batches,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "markdown_folder"
    )

    parser.add_argument(
        "output_folder"
    )

    parser.add_argument(
        "--prompt-file",
        default=str(
            Path(__file__).resolve().parent.parent
            / "prompts" / "stage3c_prompt.txt"
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N papers (in article-id order)"
    )

    args = parser.parse_args()

    api_key = os.environ.get("LLM_ROUTER_API_KEY")
    print(api_key)

    if not api_key:
        raise RuntimeError("LLM_ROUTER_API_KEY not set")

    markdown_folder = Path(args.markdown_folder)
    output_folder = Path(args.output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    log_file = output_folder / "run_log.csv"

    prompt_text = load_prompt(Path(args.prompt_file))

    paper_groups = group_markdown_by_article_id(markdown_folder)

    selected_papers = sorted(paper_groups.items())

    if args.limit is not None:
        selected_papers = selected_papers[:args.limit]

    print(
        f"Found {len(paper_groups)} papers, "
        f"processing {len(selected_papers)}"
    )

    for article_id, group in selected_papers:

        try:
            print(f"Processing {article_id}...")
            result = process_paper(
                article_id=article_id,
                markdown_files=group["markdown"],
                image_files=group["images"],
                output_root=output_folder,
                prompt_text=prompt_text,
                model=args.model,
                api_key=api_key,
            )

            if result is None:
                continue

            append_log(
                log_file=log_file,
                paper_id=result["paper_id"],
                eligible=result["eligible"],
                status=result["status"],
                message=result["message"],
                mitigation_rows=result["mitigation_rows"],
                cost_rows=result["cost_rows"],
                emission_rows=result["emission_rows"],
                runtime_seconds=result["runtime_seconds"],
                document_mode=result["document_mode"],
                markdown_count=result["markdown_count"],
                page_count=result["page_count"],
                image_count=result["image_count"],
                image_batches=result["image_batches"],
            )

        except Exception as exc:

            append_log(
                log_file=log_file,
                paper_id=article_id,
                eligible="",
                status="error",
                message=str(exc),
                runtime_seconds="",
            )

            print(f"ERROR {article_id}: {exc}")


if __name__ == "__main__":
    main()
