# stage3-extraction.py
#
# INPUT:
#     138-some-paper.pdf
#
# OUTPUT:
#
# output/
# └── 138/
#     ├── 138-mitigation.csv
#     ├── 138-costs.csv
#     └── 138-emissions.csv
#
# REQUIREMENTS:
#
# pip install pymupdf
#
# PROMPT:
#
# stage3_prompt.txt
#
# USAGE:
#
# LLM_ROUTER_API_KEY=<KEY> python3 stage3-extraction.py \
# papers \
# output
#

from pathlib import Path
from PIL import Image
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

import fitz


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"

MODE = "single_call"

PAGE_RENDER_DPI = 300

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


# ============================================================================
# PDF RENDERING
# ============================================================================

def render_pages(
    pdf_path: Path,
    render_dir: Path,
    dpi: int = PAGE_RENDER_DPI
):

    doc = fitz.open(pdf_path)

    image_paths = []

    for page_idx in range(len(doc)):

        page = doc[page_idx]

        pix = page.get_pixmap(
            dpi=dpi,
            alpha=False
        )

        output_file = (
            render_dir
            / f"page_{page_idx + 1:03d}.png"
        )

        pix.save(output_file)

        img = Image.open(output_file)

        max_dimension = max(img.size)

        if max_dimension > 2000:

            scale = 2000 / max_dimension

            new_size = (
                int(img.size[0] * scale),
                int(img.size[1] * scale),
            )

            img = img.resize(
                new_size,
                Image.LANCZOS,
            )

            img.save(output_file)
            """
            debug(
                f"Downscaled {output_file.name}: "
                f"{img.size[0]}x{img.size[1]}"
            )
            """
        """
        debug(
            f"{output_file.name}: "
            f"{img.size[0]}x{img.size[1]}"
        )
        """

        image_paths.append(output_file)


    doc.close()

    return image_paths


# ============================================================================
# OPENROUTER
# ============================================================================

def image_to_base64(
    image_path: Path
):

    with open(
        image_path,
        "rb"
    ) as f:

        return base64.b64encode(
            f.read()
        ).decode("utf-8")


def call_openrouter_vision(
    image_paths,
    prompt,
    model,
    api_key,
):

    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for image_path in image_paths:

        image_b64 = image_to_base64(
            image_path
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url":
                        f"data:image/png;base64,{image_b64}"
                }
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
            "HTTP-Referer":
                "stage3-extraction",
            "X-Title":
                "stage3-extraction",
        },
        method="POST"
    )

    last_error = None

    for attempt in range(
        1,
        RETRIES + 1
    ):

        try:

            with urllib.request.urlopen(
                request,
                timeout=180
            ) as response:

                payload = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
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
                time.sleep(
                    RETRY_SECONDS
                )
                continue

            raise last_error

        except Exception as exc:

            last_error = exc

            if attempt < RETRIES:
                time.sleep(
                    RETRY_SECONDS
                )
                continue

            raise

    raise last_error


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
        ])

def process_pdf(
    pdf_path: Path,
    output_root: Path,
    prompt_text: str,
    model: str,
    api_key: str,
):

    article_id = extract_article_id(
        pdf_path
    )

    start_time = time.time()

    debug(
        f"Processing {article_id}"
    )

    article_output_dir = (
        output_root / article_id
    )

    article_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{article_id}_"
        )
    )

    try:

        image_paths = render_pages(
            pdf_path,
            temp_dir
        )

        debug(
            f"Rendered {len(image_paths)} pages"
        )

        raw_response = (
            call_openrouter_vision(
                image_paths=image_paths,
                prompt=prompt_text,
                model=model,
                api_key=api_key,
            )
        )

        result = parse_llm_output(
            raw_response
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
        }

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


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
        default="stage3_prompt.txt"
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

    pdf_files = sorted(
        pdf_folder.glob("*.pdf")
    )

    print(
        f"Found {len(pdf_files)} PDFs"
    )

    for pdf_path in pdf_files:

        try:

            result = process_pdf(
                pdf_path=pdf_path,
                output_root=output_folder,
                prompt_text=prompt_text,
                model=args.model,
                api_key=api_key,
            )

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
            )

        except Exception as exc:

            paper_id = extract_article_id(
                pdf_path
            )

            append_log(
                log_file=log_file,
                paper_id=paper_id,
                eligible="",
                status="error",
                message=str(exc),
                runtime_seconds="",
            )

            print(
                f"ERROR {pdf_path.name}: {exc}"
            )


if __name__ == "__main__":
    main()