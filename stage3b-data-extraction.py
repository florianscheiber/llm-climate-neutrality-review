# stage3b-data-extraction.py
#
# INPUT:
#     138-main.pdf
#     138-supplement-1.pdf
#     138-supplement-2.pdf
#
# OUTPUT:
#
# output/
# └── 138/
#     ├── 138-mitigation.csv
#     ├── 138-costs.csv
#     ├── 138-emissions.csv
#     ├── 138-candidates.txt
#     ├── 138_document_content.txt
#     └── 138_raw_response.txt
#
#
# APPROACH:
#
# Stage 03b does NOT use vision or image rendering.
#
# Information is extracted from:
#
# - PDF text
# - PDF tables detected by PyMuPDF
# - figure captions contained in the PDF text
# - table captions contained in the PDF text
#
# Quantitative data are extracted only from:
#
# - text
# - machine-readable tables
#
# Figures are NOT numerically extracted.
#
# Instead, potentially relevant figures and non-extractable tables
# are reported as candidates for manual review in:
#
#     <paper_id>-candidates.txt
#
# REQUIREMENTS:
#
# pip install pymupdf
# pip install truststore
#
# PROMPT:
#
# stage3b_prompt.txt
#
# USAGE IN TERMINAL:
#
# export LLM_ROUTER_API_KEY=<KEY>
#
# python3 stage3b-data-extraction.py \
# papers-directory \
# output-directory
#
# EXAMPLE:
#
# export LLM_ROUTER_API_KEY=<KEY>
#
# python3 stage3b-data-extraction.py \
# inputs/included-after-stage-3-100 \
# outputs/output-stage3b
#
# NOTES:
#
# - PDFs belonging to the same article ID are processed together.
# - Main paper and supplementary PDFs are merged into a single document context.
# - Already processed papers are skipped automatically if all three
#   output CSVs already exist.
#

from pathlib import Path
import argparse
import csv
import json
import os
import re
import time
import urllib.request
import urllib.error

import fitz

import truststore

truststore.inject_into_ssl()



# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
MODE = "single_call"
MAX_TOKENS = 30000
RETRIES = 3
RETRY_SECONDS = 5


# ============================================================================
# HELPERS
# ============================================================================

def debug(msg):

    if os.environ.get("LLM_DEBUG") == "1":
        print(f"[DEBUG] {msg}")


def extract_article_id(pdf_path: Path) -> str:

    match = re.match(
        r"^(\d+)",
        pdf_path.stem
    )

    if not match:
        raise RuntimeError(
            f"Could not extract article id from {pdf_path.name}"
        )

    return match.group(1)


def load_prompt(prompt_path: Path) -> str:

    with open(
        prompt_path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()

def group_pdfs_by_article_id(pdf_folder):

    groups = {}

    for pdf_path in pdf_folder.glob("*.pdf"):

        article_id = extract_article_id(
            pdf_path
        )

        groups.setdefault(
            article_id,
            []
        ).append(pdf_path)

    return groups

def extract_document_content(
    pdf_paths
):

    parts = []

    for pdf_path in pdf_paths:

        parts.append(
            f"\n\n====================\n"
            f"DOCUMENT: {pdf_path.name}\n"
            f"====================\n"
        )

        doc = fitz.open(pdf_path)

        for page_idx in range(len(doc)):

            page = doc[page_idx]

            page_text = page.get_text()

            if page_text.strip():

                parts.append(
                    f"\n[PAGE {page_idx + 1} TEXT]\n"
                )

                parts.append(page_text)

            try:

                tables = page.find_tables()

                if tables.tables:

                    for table_idx, table in enumerate(
                        tables.tables,
                        start=1
                    ):

                        parts.append(
                            f"\n[PAGE {page_idx + 1} "
                            f"TABLE {table_idx}]\n"
                        )

                        extracted = table.extract()

                        for row in extracted:

                            row_text = " | ".join(
                                str(cell)
                                if cell is not None
                                else ""
                                for cell in row
                            )

                            parts.append(
                                row_text
                            )

            except Exception:

                pass

        doc.close()

    return "\n".join(parts)

# ============================================================================
# OPENROUTER
# ============================================================================


def call_openrouter_text(
    document_content,
    prompt,
    model,
    api_key,
):

    content = (
        prompt
        + "\n\n"
        + document_content
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

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        url="https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization":
                f"Bearer {api_key}",
            "Content-Type":
                "application/json",
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=180
    ) as response:

        payload = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

        return (
            payload["choices"][0]
            ["message"]
            ["content"]
        )

# ============================================================================
# JSON
# ============================================================================

def parse_llm_output(
    raw_text
):

    raw_text = raw_text.strip()

    if not raw_text.startswith("{"):

        start = raw_text.find("{")
        end = raw_text.rfind("}")

        if start != -1 and end != -1:

            raw_text = (
                raw_text[start:end + 1]
            )

    return json.loads(raw_text)


def normalize_output_rows(result):

    for section_name in [
        "mitigation",
        "costs",
        "emissions",
    ]:

        rows = result.get(
            section_name,
            []
        )

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
                normalized_row["Area"] = (
                    normalized_row["Region"]
                )

            if (
                not normalized_row.get("Source_Number")
                and normalized_row.get("Source_Detail")
            ):
                normalized_row["Source_Number"] = (
                    normalized_row["Source_Detail"]
                )

            normalized_rows.append(
                normalized_row
            )

        result[section_name] = normalized_rows

    return result


# ============================================================================
# CSV WRITING
# ============================================================================

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
    "Variable_Cost",
    "Variable_Cost_Unit",
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


def write_csv(
    path,
    columns,
    rows
):

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

                clean_row[col] = (
                    row.get(col, "")
                )

            writer.writerow(
                clean_row
            )


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
    pdf_count="",
    page_count="",
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
                "pdf_count",
                "page_count",
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
            pdf_count,
            page_count,
        ])

def process_paper(
    article_id,
    pdf_paths,
    output_root: Path,
    prompt_text: str,
    model: str,
    api_key: str,
):
    start_time = time.time()

    debug(
        f"Processing {article_id}"
    )

    article_output_dir = (
            output_root / article_id
    )

    mitigation_csv = (
            article_output_dir
            / f"{article_id}-mitigation.csv"
    )

    costs_csv = (
            article_output_dir
            / f"{article_id}-costs.csv"
    )

    emissions_csv = (
            article_output_dir
            / f"{article_id}-emissions.csv"
    )

    if (
            mitigation_csv.exists()
            and costs_csv.exists()
            and emissions_csv.exists()
    ):
        print(
            f"{article_id}: already processed"
        )

        return None

    article_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    main_pdfs = [
        p for p in pdf_paths
        if "-main.pdf" in p.name
    ]

    if len(main_pdfs) != 1:
        raise RuntimeError(
            "no unique main pdf found"
        )

    main_pdf = main_pdfs[0]

    supplement_pdfs = [
        p for p in pdf_paths
        if p != main_pdf
    ]

    selected_pdfs = (
            [main_pdf]
            + supplement_pdfs
    )

    document_mode = (
        "main_plus_supplements"
    )

    pdf_count = len(
        selected_pdfs
    )

    page_count = 0

    for pdf in selected_pdfs:
        doc = fitz.open(pdf)

        page_count += len(doc)

        doc.close()

    document_content = (
        extract_document_content(
            selected_pdfs
        )
    )

    with open(
            article_output_dir
            / f"{article_id}_document_content.txt",
            "w",
            encoding="utf-8"
    ) as f:

        f.write(
            document_content
        )

    raw_response = (
        call_openrouter_text(
            document_content=document_content,
            prompt=prompt_text,
            model=model,
            api_key=api_key,
        )
    )

    with open(
            article_output_dir / f"{article_id}_raw_response.txt",
            "w",
            encoding="utf-8"
    ) as f:
        f.write(raw_response)

    result = parse_llm_output(
        raw_response
    )

    result = normalize_output_rows(
        result
    )

    candidate_file = (
            article_output_dir
            / f"{article_id}-candidates.txt"
    )

    with open(
            candidate_file,
            "w",
            encoding="utf-8"
    ) as f:

        for candidate in result.get(
                "candidates",
                []
        ):
            f.write(
                f"{candidate.get('identifier', '')}\n"
                f"Reason: {candidate.get('reason', '')}\n"
                f"Expected information type: "
                f"{candidate.get('expected_information_type', '')}\n\n"
            )

    eligible = result.get(
        "eligible",
        True
    )

    mitigation_rows = result.get(
        "mitigation",
        []
    )

    cost_rows = result.get(
        "costs",
        []
    )

    emission_rows = result.get(
        "emissions",
        []
    )

    mitigation_csv = (
        article_output_dir
        / f"{article_id}-mitigation.csv"
    )

    costs_csv = (
        article_output_dir
        / f"{article_id}-costs.csv"
    )

    emissions_csv = (
        article_output_dir
        / f"{article_id}-emissions.csv"
    )

    if not eligible:

        write_csv(
            mitigation_csv,
            MITIGATION_COLUMNS,
            [],
        )

        write_csv(
            costs_csv,
            COST_COLUMNS,
            [],
        )

        write_csv(
            emissions_csv,
            EMISSION_COLUMNS,
            [],
        )

        print(
            f"{article_id}: not eligible"
        )

        runtime_seconds = round(
            time.time() - start_time,
            1
        )

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
            "pdf_count": pdf_count,
            "page_count": page_count,
        }

    write_csv(
        mitigation_csv,
        MITIGATION_COLUMNS,
        mitigation_rows,
    )

    write_csv(
        costs_csv,
        COST_COLUMNS,
        cost_rows,
    )

    write_csv(
        emissions_csv,
        EMISSION_COLUMNS,
        emission_rows,
    )

    print(
        f"{article_id}: completed"
    )

    runtime_seconds = round(
        time.time() - start_time,
        1
    )

    return {
        "paper_id": article_id,
        "eligible": True,
        "status": "saved",
        "message": "ok",
        "mitigation_rows": len(
            mitigation_rows
        ),
        "cost_rows": len(
            cost_rows
        ),
        "emission_rows": len(
            emission_rows
        ),
        "runtime_seconds": runtime_seconds,
        "document_mode": document_mode,
        "pdf_count": pdf_count,
        "page_count": page_count,
    }

# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pdf_folder"
    )

    parser.add_argument(
        "output_folder"
    )

    parser.add_argument(
        "--prompt-file",
        default="stage3b_prompt.txt"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL
    )

    args = parser.parse_args()

    api_key = os.environ.get(
        "LLM_ROUTER_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "LLM_ROUTER_API_KEY not set"
        )

    pdf_folder = Path(
        args.pdf_folder
    )

    output_folder = Path(
        args.output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    log_file = (
            output_folder
            / "run_log.csv"
    )

    prompt_text = load_prompt(
        Path(args.prompt_file)
    )

    paper_groups = (
        group_pdfs_by_article_id(
            pdf_folder
        )
    )

    print(
        f"Found {len(paper_groups)} papers"
    )

    for article_id, pdf_paths in sorted(
            paper_groups.items()
    ):

        try:

            result = process_paper(
                article_id=article_id,
                pdf_paths=pdf_paths,
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
                pdf_count=result["pdf_count"],
                page_count=result["page_count"],
            )

        except Exception as exc:

            paper_id = article_id

            append_log(
                log_file=log_file,
                paper_id=paper_id,
                eligible="",
                status="error",
                message=str(exc),
                runtime_seconds="",
            )

            print(
                f"ERROR {article_id}: {exc}"
            )


if __name__ == "__main__":
    main()