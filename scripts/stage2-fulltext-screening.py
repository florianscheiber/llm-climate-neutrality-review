"""
aktuelle Herausforderungen:
- Sind alle pdf gesammelt verfügbar? lokale ablage oder cloud?
- wie verknüpfe ich pdf mit doi? Im Dateinamen oder mapping datei (input csv mit pdf name und doi)?


TO DOs:
- check if i can download pdfs
- test set (3 paper) erstellen, um zu sehen ob es funktioniert und wie viel es kostet
- Bei den ersten 3 PDFs unbedingt die ersten ~3000 extrahierten Zeichen ausgeben und manuell prüfen.


Remarks:
- der generelle Aufbau zum abstract screening bleibt gleich
- nicht die ganze pdf einlesen sondern References, Appendix, Acknowledgements ... wegcutten --> das im API call umsetzen
- mindestanzahl an wörtern die erkannt werden, sonst wird das paper rausgeschmissen
- cut off after references

papers_test/
├── paper1.pdf
├── paper2.pdf
└── paper3.pdf

pdf-screening.py

Aufruf:

LLM_ROUTER_API_KEY=<key> python3 stage2-fulltext-screening.py \
papers_directory \
output.csv \
--provider router \
--model anthropic/claude-sonnet-4.6


credit for run1: 0,94 dollar

"""

from typing import Dict, Optional
from pathlib import Path
import json
import os
import argparse
import csv
import time
import urllib.request
import urllib.error
import sys

import fitz


# ============================================================================
# CONFIG
# ============================================================================

RESULT_COLUMNS = [
    "paper_id",
    "exclude_reason",
    "exclude_comment",
    "q1_sector",
    "q1_comment",
    "q1_evidence",
    "q2_quantitative_pathway",
    "q2_comment",
    "q2_evidence",
]

VALID_LABELS = {"yes", "no"}

VALID_EXCLUDE_REASONS = {
    "none",
    "fulltext_not_available",
    "fulltext_not_accessible",
    "duplicate",
    "no_primary_data",
}

REFERENCE_HEADERS = [
    "references",
    "bibliography",
    "works cited",
    "literature",
]

MIN_TEXT_LENGTH = 1000

LLM_PROVIDER_SETTINGS = {
    "router": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_envs": ["LLM_ROUTER_API_KEY", "OPENROUTER_API_KEY"],
    }
}

# ============================================================================
# PROMPT
# ============================================================================

LLM_PROMPT = """
You are a scientific full-text screening assistant.

Evaluate EXACTLY ONE scientific paper.

Return ONLY a valid JSON object.

IMPORTANT:

The following exclusion categories are available:

- none
- fulltext_not_available
- fulltext_not_accessible
- duplicate
- no_primary_data

If assessment is possible:
- exclude_reason = "none"
- answer YES or NO only
- do not use unclear

QUESTION 1

Does the study explicitly include at least one of the following sectors
within its system boundary?

AGRICULTURE
ENERGY
INDUSTRY
LAND-USE
FORESTRY
WASTE
CROSS-SECTORAL
OTHER

Rules:

- CROSS-SECTORAL only if sectors are treated as an integrated whole-economy system.
- If yes:
  q1_comment must contain all applicable labels separated by commas.
- If no:
  q1_comment must be empty.

Provide a short evidence quote/paraphrase.

QUESTION 2

Does the study present at least one quantitative scenario or pathway
toward climate neutrality?

Focus primarily on:
- Results
- Results and Discussion
- Figures
- Figure captions
- Tables
- Table captions

If yes:

Use one or more labels:

DEMAND
EMISSIONS
COSTS
PRICES
TECHNOLOGY-SHARES
LAND-USE REQUIREMENTS
OTHER

If no:
leave q2_comment empty.

Provide a short evidence quote/paraphrase.

Return EXACTLY:

{
  "exclude_reason": "...",
  "exclude_comment": "...",

  "q1_sector": "yes/no",
  "q1_comment": "...",
  "q1_evidence": "...",

  "q2_quantitative_pathway": "yes/no",
  "q2_comment": "...",
  "q2_evidence": "..."
}
"""


# ============================================================================
# HELPERS
# ============================================================================

def _debug_log(message: str) -> None:
    if os.environ.get("LLM_DEBUG") == "1":
        print(f"[DEBUG] {message}", file=sys.stderr)


def _short_error(exc: Exception, max_len: int = 200) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    return (msg[:max_len] + "...") if len(msg) > max_len else msg


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 429} or 500 <= status_code < 600


# ============================================================================
# PDF
# ============================================================================

def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)

    pages = []

    for page in doc:
        pages.append(page.get_text())

    doc.close()

    return "\n".join(pages)


def remove_references(pdf_text: str) -> str:
    lower_text = pdf_text.lower()

    for marker in REFERENCE_HEADERS:
        pos = lower_text.find(marker)

        if pos != -1:
            return pdf_text[:pos]

    return pdf_text


def validate_pdf_text(pdf_text: str) -> None:
    if len(pdf_text.strip()) < MIN_TEXT_LENGTH:
        raise RuntimeError(
            "PDF contains too little extractable text. OCR may be required."
        )


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_result(raw_result: Dict) -> Dict:
    result = {}

    exclude_reason = str(
        raw_result.get("exclude_reason", "none")
    ).strip()

    if exclude_reason not in VALID_EXCLUDE_REASONS:
        exclude_reason = "fulltext_not_accessible"

    result["exclude_reason"] = exclude_reason
    result["exclude_comment"] = str(
        raw_result.get("exclude_comment", "")
    ).strip()

    for field in [
        "q1_sector",
        "q2_quantitative_pathway",
    ]:
        value = str(raw_result.get(field, "")).strip().lower()

        if value not in VALID_LABELS:
            value = "no"

        result[field] = value

    for field in [
        "q1_comment",
        "q1_evidence",
        "q2_comment",
        "q2_evidence",
    ]:
        result[field] = str(
            raw_result.get(field, "")
        ).strip()

    return result


# ============================================================================
# API CLIENT
# ============================================================================

class LLMApiClient:

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 120,
        retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay

    def chat_completions_create(
        self,
        *,
        model: str,
        temperature: float,
        response_format: Dict,
        messages: list,
    ) -> Dict:

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "temperature": temperature,
            "response_format": response_format,
            "messages": messages,
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "fulltext-screening",
                "X-Title": "pdf-screening",
            },
            method="POST",
        )

        last_error = None

        for attempt in range(1, self.retries + 1):

            try:
                with urllib.request.urlopen(
                    req,
                    timeout=self.timeout,
                ) as resp:
                    return json.loads(
                        resp.read().decode("utf-8")
                    )

            except urllib.error.HTTPError as exc:

                body = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )

                last_error = RuntimeError(
                    f"LLM HTTP {exc.code}: {body}"
                )

                if (
                    attempt < self.retries
                    and _is_retryable_status(exc.code)
                ):
                    time.sleep(self.retry_delay)
                    continue

                raise last_error

            except Exception as exc:

                last_error = RuntimeError(
                    f"LLM request failed: {exc}"
                )

                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                    continue

                raise last_error

        raise last_error


# ============================================================================
# CONFIG
# ============================================================================

def resolve_llm_key(
    provider_id: str,
    direct_key: Optional[str],
) -> str:

    if direct_key:
        return direct_key

    cfg = LLM_PROVIDER_SETTINGS[provider_id]

    for env_name in [
        "LLM_API_KEY",
        "LLM_KEY",
        *cfg["key_envs"],
    ]:
        value = os.environ.get(env_name)

        if value:
            return value

    raise RuntimeError("No API key found")


def build_llm_client(
    provider_id: str,
    api_key: str,
):

    cfg = LLM_PROVIDER_SETTINGS[provider_id]

    return LLMApiClient(
        api_key=api_key,
        base_url=cfg["base_url"],
    )


# ============================================================================
# LLM CALL
# ============================================================================

def call_llm_once(
    client: LLMApiClient,
    model: str,
    pdf_text: str,
) -> Dict:

    response = client.chat_completions_create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": LLM_PROMPT,
            },
            {
                "role": "user",
                "content": pdf_text,
            },
        ],
    )

    content = (
        (response.get("choices") or [{}])[0]
        .get("message", {})
        .get("content")
    )

    if not content:
        raise RuntimeError("Empty response")

    content = content.strip()

    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1:
            content = content[start:end + 1]

    parsed = json.loads(content)

    return normalize_result(parsed)


# ============================================================================
# PIPELINE
# ============================================================================

def process_pdf_folder(
    pdf_folder: str,
    output_csv: str,
    model: str,
    provider_id: str,
    api_key: Optional[str],
):

    resolved_key = resolve_llm_key(
        provider_id,
        api_key,
    )

    client = build_llm_client(
        provider_id,
        resolved_key,
    )

    pdf_files = sorted(
        Path(pdf_folder).glob("*.pdf")
    )

    print("pdf_folder =", pdf_folder)
    print("absolute path =", Path(pdf_folder).resolve())
    print("exists =", Path(pdf_folder).exists())
    print("is_dir =", Path(pdf_folder).is_dir())

    with open(
        output_csv,
        "w",
        encoding="utf-8",
        newline=""
    ) as fout:

        writer = csv.DictWriter(
            fout,
            fieldnames=RESULT_COLUMNS,
            delimiter=";"
        )

        writer.writeheader()

        for idx, pdf_path in enumerate(pdf_files, start=1):

            paper_id = pdf_path.stem

            try:

                pdf_text = extract_pdf_text(
                    str(pdf_path)
                )

                validate_pdf_text(pdf_text)

                pdf_text = remove_references(
                    pdf_text
                )

                if os.environ.get("LLM_DEBUG") == "1" and idx <= 3:
                    print("=" * 80)
                    print(paper_id)
                    print(pdf_text[:3000])
                    print("=" * 80)

                result = call_llm_once(
                    client,
                    model,
                    pdf_text,
                )

            except Exception as exc:

                err = _short_error(exc)

                _debug_log(err)

                result = {
                    "exclude_reason":
                        "fulltext_not_accessible",
                    "exclude_comment":
                        err,
                    "q1_sector":
                        "no",
                    "q1_comment":
                        "",
                    "q1_evidence":
                        "",
                    "q2_quantitative_pathway":
                        "no",
                    "q2_comment":
                        "",
                    "q2_evidence":
                        "",
                }

            writer.writerow(
                {
                    "paper_id": paper_id,
                    **result,
                }
            )

            print(f"Processed: {paper_id}")


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Screen PDFs with OpenRouter"
    )

    parser.add_argument(
        "pdf_folder"
    )

    parser.add_argument(
        "output_csv"
    )

    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4.6",
    )

    parser.add_argument(
        "--provider",
        default="router",
        choices=["router"],
    )

    parser.add_argument(
        "--api-key",
        default=None,
    )

    args = parser.parse_args()

    process_pdf_folder(
        args.pdf_folder,
        args.output_csv,
        args.model,
        args.provider,
        args.api_key,
    )


if __name__ == "__main__":
    main()