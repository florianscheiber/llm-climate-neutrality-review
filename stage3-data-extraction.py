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

MAX_TOKENS = 20000

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
                timeout=300
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

            return

        write_csv(
            mitigation_csv,
            MITIGATION_COLUMNS,
            result.get(
                "mitigation",
                []
            ),
        )

        write_csv(
            costs_csv,
            COST_COLUMNS,
            result.get(
                "costs",
                []
            ),
        )

        write_csv(
            emissions_csv,
            EMISSION_COLUMNS,
            result.get(
                "emissions",
                []
            ),
        )

        print(
            f"{article_id}: completed"
        )

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

            process_pdf(
                pdf_path=pdf_path,
                output_root=output_folder,
                prompt_text=prompt_text,
                model=args.model,
                api_key=api_key,
            )

        except Exception as exc:

            print(
                f"ERROR {pdf_path.name}: {exc}"
            )


if __name__ == "__main__":
    main()