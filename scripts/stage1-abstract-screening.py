# ============================================================================
# IMPORTS
# ============================================================================

from typing import Dict, Optional
import json
import os
import argparse
import csv
import time
import urllib.request
import urllib.error
import sys

"""
AUTOMATED ABSTRACT SCREENING PIPELINE FOR CLIMATE NEUTRALITY REVIEW

PURPOSE:
  Screen ~600 scientific paper abstracts using an LLM API to answer 4 predefined questions.
  Results are compared with human reviewers (CADIMA) for validation.

SCREENING QUESTIONS:
  Q1: Does the abstract describe a pathway or system that is climate-neutral?
      (Terms like "net zero" count ONLY if actually assessed, not just mentioned)
  Q2: Does the abstract target at least one specific region? (Global-only = no)
  Q3: Does the abstract address at least one sector? (energy, industry, transport, etc.)
  Q4: Does the abstract include quantitative modelling or scenario analysis?

OUTPUT SCHEMA:
  For each question: answer must be "yes", "no", or "unclear"
  Comments: only required if answer is "unclear", otherwise empty

REPRODUCIBILITY:
  - Temperature = 0 (deterministic)
  - Each paper processed independently
  - JSON schema enforcement ensures valid outputs
"""

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Expected output columns in the results CSV
RESULT_COLUMNS = [
    "doi",
    "q1_climate_neutrality",
    "q1_comment",
    "q2_region",
    "q2_comment",
    "q3_sector",
    "q3_comment",
    "q4_method",
    "q4_comment",
]

# Mapping of (question_field, comment_field) pairs for validation/normalization
RESULT_FIELDS = [
    ("q1_climate_neutrality", "q1_comment"),
    ("q2_region", "q2_comment"),
    ("q3_sector", "q3_comment"),
    ("q4_method", "q4_comment"),
]

# Valid answer labels for any question
VALID_LABELS = {"yes", "no", "unclear"}

# Provider configurations: base URLs and environment variable names for API keys
# This allows flexibility to switch between OpenAI, OpenRouter, Groq, Together, or custom endpoints
LLM_PROVIDER_SETTINGS = {
    "default": {
        "base_url": None,
        "key_envs": ["LLM_DEFAULT_API_KEY"],
    },
    "router": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_envs": ["LLM_ROUTER_API_KEY", "OPENROUTER_API_KEY"],
    },
    "speed": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_envs": ["LLM_SPEED_API_KEY", "GROQ_API_KEY"],
    },
    "cloud": {
        "base_url": "https://api.together.xyz/v1",
        "key_envs": ["LLM_CLOUD_API_KEY", "TOGETHER_API_KEY"],
    },
}

# ============================================================================
# SCREENING PROMPT (System message for LLM)
# ============================================================================
LLM_PROMPT = """
You are a scientific abstract screening assistant for a climate neutrality review.

Your task: Answer exactly 4 questions about each abstract and title using ONLY the information provided.
Return a JSON object with 8 fields (4 questions + 4 comments).

CRITICAL RULES:
Base decisions ONLY on the abstract text AND the title. Use NO external knowledge.
Each paper is independent. Do NOT cross-reference with other papers.
For each question, respond with EXACTLY one of: "yes", "no", "unclear"
If given Information is not enough for evaluating the question, respond with "unclear" and comment.
ONLY COMMENT IF your answer is "unclear" (otherwise leave empty).
Comments must be SHORT (max 1-2 sentences explaining why uncertain).
Directly stick to the informations provided.
DO NOT guess any answers. DO NOT invent any answers or information.

QUESTION DEFINITIONS:
Q1 (Climate Neutrality):

Does the abstract provide a pathway or a system configuration which IS climate-neutral?
We also accept "net zero", "carbon neutral(ity)", "zero emissions/carbon", "deep decarbonization" and similar
Merely mentioning "net zero" is NOT sufficient. It must be actually assessed/modeled.
Answer: "yes" if pathway/config is climate-neutral | "no" if not | "unclear" if ambiguous
Q2 (Region):

Does the abstract target at least ONE specific spatial area?
OK: (group of) country, city, state, province, district
"Global" or "worldwide" analysis alone = "no"
Answer: "yes" if specific region(s) mentioned | "no" if global-only | "unclear" if ambiguous
Q3 (Sector):

Does the abstract target at least ONE of the following sectors:
energy, industry, agriculture, forestry, land use, or AFOLU (agriculture, forestry and other land uses)
Answer: "yes" if sector specified | "no" if none mentioned | "unclear" if unclear
Q4 (Method):

Does the abstract include quantitative modelling and/or a scenario process to generate climate neutral pathways or systems?
Qualitative discussion alone = "no"
Answer: "yes" if quantitative/modeling present | "no" if not | "unclear" if unclear
RESPONSE FORMAT (JSON):
{
  "q1_climate_neutrality": "yes|no|unclear",
  "q1_comment": "explanation if unclear, else empty string",
  "q2_region": "yes|no|unclear",
  "q2_comment": "explanation if unclear, else empty string",
  "q3_sector": "yes|no|unclear",
  "q3_comment": "explanation if unclear, else empty string",
  "q4_method": "yes|no|unclear",
  "q4_comment": "explanation if unclear, else empty string"
}

"""

# ============================================================================
# NORMALIZATION FUNCTION
# ============================================================================

def normalize_result(raw_result: Dict[str, str]) -> Dict[str, str]:
    """
    Validate and normalize LLM output to ensure it strictly follows the schema.

    INPUT:
      raw_result: Dict with keys like "q1_climate_neutrality", "q1_comment", etc.
                  (May contain invalid labels, malformed JSON, typos, etc.)

    OUTPUT:
      normalized: Dict with same keys, guaranteed to contain only:
                  - Labels in {"yes", "no", "unclear"}
                  - Comments: empty string if label is "yes"/"no",
                             or short explanation if label is "unclear"

    LOGIC:
      1. For each (question, comment) pair:
         - Extract the label and convert to lowercase/stripped
         - If label is invalid/missing → set to "unclear" with error explanation
         - If label is "yes"/"no" → clear the comment (not needed)
         - If label is "unclear" and no comment provided → add default explanation
      2. This ensures robustness against malformed LLM outputs

    RISK MITIGATION:
      - Invalid JSON from LLM → fallback to "unclear"
      - Missing fields → fallback to "unclear"
      - Typos ("YES", "No.", etc.) → normalized to valid values
    """
    normalized: Dict[str, str] = {}

    for label_key, note_key in RESULT_FIELDS:
        # Extract the raw label and comment from LLM output
        label = str(raw_result.get(label_key, "")).strip().lower()
        note = str(raw_result.get(note_key, "")).strip()

        # VALIDATION: If label is not one of the allowed values, mark as "unclear"
        if label not in VALID_LABELS:
            label = "unclear"
            note = "invalid or missing model output"

        # NORMALIZATION: Clear comments for "yes"/"no" answers (not needed per spec)
        if label in {"yes", "no"}:
            note = ""
        # If unclear but no comment provided, add default explanation
        elif not note:
            note = "insufficient information in abstract"

        normalized[label_key] = label
        normalized[note_key] = note

    return normalized

# ============================================================================
# LLM API CLIENT (HTTP wrapper for OpenAI-compatible APIs)
# ============================================================================

class LLMApiClient:
    """
    Low-level HTTP client for communicating with OpenAI-compatible LLM endpoints.
    """
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 60,
        retries: int = 3,
        retry_delay: float = 1.5,
    ) -> None:
        # API-Schlüssel + Endpoint werden zentral am Client gehalten.
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Kleine Retry-Policy für Timeouts, 429 und temporäre Serverfehler.
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
        """
        Make a single chat completions API call with simple retry logic.
        """
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
                # OpenRouter-empfohlene optionale Metadaten-Header
                "HTTP-Referer": "climate-review-project",
                "X-Title": "abstract-screening",
            },
            method="POST",
        )

        last_error: Optional[Exception] = None

        # Mehrfach versuchen, damit kurze Störungen nicht sofort zu "unclear" führen.
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"LLM HTTP {exc.code}: {body}")

                # Nur temporäre HTTP-Fehler erneut versuchen.
                if attempt < self.retries and _is_retryable_status(exc.code):
                    time.sleep(self.retry_delay)
                    continue
                raise last_error from exc

            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"LLM network error: {exc}")
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                    continue
                raise last_error from exc

            except json.JSONDecodeError as exc:
                # Falls die Antwort kurzfristig kaputt ist, kann ein Retry helfen.
                last_error = RuntimeError(f"LLM invalid JSON response: {exc}")
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                    continue
                raise last_error from exc

        # Sicherheitsnetz; wird praktisch nur erreicht, wenn alle Versuche scheitern.
        raise last_error or RuntimeError("LLM request failed")

# ============================================================================
# RETRY-HELPER
# ============================================================================
def _is_retryable_status(status_code: int) -> bool:
    """
    Entscheidet, ob ein HTTP-Status ein Retry rechtfertigt.
    - 408/409/429: Timeout/Konflikt/Rate Limit
    - 5xx: temporäre Serverfehler
    """
    return status_code in {408, 409, 429} or 500 <= status_code < 600

# ============================================================================
# MAIN SCREENING FUNCTION (Single paper → LLM call → structured JSON)
# ============================================================================

def call_llm_once(client: LLMApiClient, model: str, abstract: str) -> Dict[str, str]:
    """
    Send one abstract to the LLM and get structured screening results.
    """
    response = client.chat_completions_create(
        model=model,
        temperature=0,
        # OpenRouter-kompatibel: kein strict json_schema, nur json_object.
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": f"Abstract:\n{abstract or ''}"},
        ],
    )

    content = (
        (response.get("choices") or [{}])[0]
        .get("message", {})
        .get("content")
    )

    if not content:
        raise RuntimeError(f"Empty response: {response}")

    content = content.strip()

    # JSON extrahieren (fallback falls Claude Text davor schreibt)
    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            content = content[start:end + 1]

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON from model: {content}")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise

    return normalize_result(parsed)

# ============================================================================
# DEBUG LOGGING (optional, controlled via env var)
# ============================================================================
def _debug_log(message: str) -> None:
    """
    Ausgabe von Debug-Infos nur wenn LLM_DEBUG=1 gesetzt ist.
    """
    if os.environ.get("LLM_DEBUG") == "1":
        print(f"[DEBUG] {message}", file=sys.stderr)

# ============================================================================
# CONFIGURATION RESOLUTION FUNCTIONS
# ============================================================================

def resolve_llm_key(provider_id: str, direct_key: Optional[str]) -> str:
    """
    Resolve the LLM API key from multiple sources in priority order.

    INPUT:
      provider_id: Provider identifier (e.g., "router", "speed", "cloud", "default")
      direct_key: API key passed directly via CLI argument (highest priority)

    RESOLUTION ORDER:
      1. If direct_key provided → use it (CLI takes precedence)
      2. Check generic env vars: LLM_API_KEY, LLM_KEY
      3. Check provider-specific env vars (from LLM_PROVIDER_SETTINGS)
      4. If none found → raise error with helpful message

    OUTPUT:
      api_key: The resolved API key string

    ERROR HANDLING:
      - If no key found anywhere → RuntimeError with list of expected env vars

    EXAMPLE:
      resolve_llm_key("router", None)
      → Checks: LLM_API_KEY, LLM_KEY, LLM_ROUTER_API_KEY, OPENROUTER_API_KEY
      → Returns first one found, or raises error
    """
    # If key provided directly via CLI, use it immediately
    if direct_key:
        return direct_key

    # Get provider configuration (or fall back to default)
    cfg = LLM_PROVIDER_SETTINGS.get(provider_id, LLM_PROVIDER_SETTINGS["default"])

    # Build list of environment variables to check in order
    candidate_envs = ["LLM_API_KEY", "LLM_KEY", *cfg["key_envs"]]

    # Check each environment variable
    for env_name in candidate_envs:
        value = os.environ.get(env_name)
        if value:
            return value

    # No key found → provide helpful error message
    raise RuntimeError(
        f"Kein LLM-Key gefunden. Setze --api-key oder eine Env-Variable aus: {', '.join(candidate_envs)}"
    )


def build_llm_client(
    provider_id: str,
    api_key: str,
    explicit_base_url: Optional[str],
) -> LLMApiClient:
    """
    Build and initialize an LLMApiClient instance with resolved configuration.

    INPUT:
      provider_id: Provider name (e.g., "router", "speed", "cloud")
      api_key: Resolved API key (from resolve_llm_key)
      explicit_base_url: Base URL from CLI (overrides provider default)

    RESOLUTION ORDER:
      1. If explicit_base_url provided → use it (CLI takes precedence)
      2. Check LLM_BASE_URL environment variable
      3. Use provider's default base_url
      4. If none → raise error

    OUTPUT:
      client: Initialized LLMApiClient instance ready for API calls

    ERROR HANDLING:
      - If no base_url found → RuntimeError with helpful message

    EXAMPLE:
      build_llm_client("router", "sk-...", None)
      → Creates client with OpenRouter base_url + API key
    """
    # Get provider configuration
    cfg = LLM_PROVIDER_SETTINGS.get(provider_id, LLM_PROVIDER_SETTINGS["default"])

    # Resolve base_url with priority: CLI argument > env var > provider default
    api_base = (
        explicit_base_url
        or os.environ.get("LLM_BASE_URL")
        or cfg["base_url"]
    )

    # Validate that we have a base_url
    if not api_base:
        raise RuntimeError(
            "Keine LLM Base-URL gesetzt. Nutze --base-url oder LLM_BASE_URL."
        )

    # Create and return the client
    return LLMApiClient(api_key=api_key, base_url=api_base)

# ============================================================================
# CSV PROCESSING PIPELINE
# ============================================================================

def process_csv(
    input_csv: str,
    output_csv: str,
    model: str,
    provider_id: str,
    llm_key: Optional[str],
    api_base: Optional[str],
) -> None:
    """
    Main pipeline: Read input CSV, call LLM for each abstract, write results.

    INPUT:
      input_csv: Path to input CSV with columns: doi, abstract
      output_csv: Path to output CSV (will be created/overwritten)
      model: Model name to use (e.g., "gpt-4o-mini")
      provider_id: Provider identifier for API endpoint selection
      llm_key: Optional API key (if not provided, resolved from env)
      api_base: Optional base URL (if not provided, resolved from env)

    PROCESS:
      1. Resolve API key from CLI/env
      2. Initialize LLM client with resolved credentials
      3. Open input CSV for reading, output CSV for writing
      4. Write CSV header with expected columns
      5. For each row in input:
         a. Extract doi and abstract
         b. Call LLM for this paper
         c. Normalize result
         d. Write to output CSV
         e. If error → fallback to all "unclear" with error message
      6. Close files

    ERROR HANDLING:
      - Missing input file → FileNotFoundError (not caught, fails loudly)
      - API errors (rate limit, invalid key) → caught, paper marked "unclear"
      - Invalid JSON from LLM → caught, paper marked "unclear"
      - Network timeouts → caught, paper marked "unclear"

    OUTPUT FILE:
      CSV with columns: doi, q1_*, q1_comment, q2_*, q2_comment, ...
      Each row = one paper with all 4 question answers

    PERFORMANCE NOTE:
      - Synchronous: processes papers one at a time
      - For 600 papers, expect ~10-30 minutes depending on API response time
      - Consider batching or async for faster processing (future enhancement)

    RISK MITIGATION:
      - Rate limits: Request will fail with HTTP 429 → caught as RuntimeError
      - Quota exceeded: Request will fail with HTTP 403 → caught as RuntimeError
      - Invalid model: Request will fail with HTTP 400 → caught as RuntimeError
    """
    # Resolve LLM credentials
    resolved_key = resolve_llm_key(provider_id, llm_key)

    # Initialize API client
    client = build_llm_client(provider_id, resolved_key, api_base)

    # Open input and output files
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as fout:
        # Create CSV reader/writer
        reader = csv.DictReader(fin, delimiter=";")
        writer = csv.DictWriter(fout, fieldnames=RESULT_COLUMNS)


        # Write output header
        writer.writeheader()

        # Process each paper
        for row in reader:
            # Extract required fields
            doi = row.get("doi", "")
            abstract = row.get("abstract", "")

            try:
                # Call LLM for this abstract
                screened = call_llm_once(client, model, abstract)
            except Exception as exc:
                # Konkrete Fehlermeldung sichtbar machen (statt nur "RuntimeError")
                err_msg = _short_error(exc)
                _debug_log(f"LLM failed: {err_msg}")
                screened = {
                    "q1_climate_neutrality": "unclear",
                    "q1_comment": f"llm error: {err_msg}",
                    "q2_region": "unclear",
                    "q2_comment": f"llm error: {err_msg}",
                    "q3_sector": "unclear",
                    "q3_comment": f"llm error: {err_msg}",
                    "q4_method": "unclear",
                    "q4_comment": f"llm error: {err_msg}",
                }

            # Write result row to output CSV
            writer.writerow(
                {
                    "doi": doi,
                    "q1_climate_neutrality": screened["q1_climate_neutrality"],
                    "q1_comment": screened["q1_comment"],
                    "q2_region": screened["q2_region"],
                    "q2_comment": screened["q2_comment"],
                    "q3_sector": screened["q3_sector"],
                    "q3_comment": screened["q3_comment"],
                    "q4_method": screened["q4_method"],
                    "q4_comment": screened["q4_comment"],
                }
            )

# ============================================================================
# ERROR HELPERS
# ============================================================================
def _short_error(exc: Exception, max_len: int = 200) -> str:
    """
    Kürzt Fehlermeldungen für CSV-Ausgabe, damit sie lesbar bleiben.
    """
    msg = f"{type(exc).__name__}: {exc}"
    return (msg[:max_len] + "...") if len(msg) > max_len else msg

# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Parse command-line arguments and launch the screening pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Screen scientific abstracts using an LLM API"
    )
    # OpenRouter-Modelle brauchen typischerweise den Provider-Prefix.
    parser.add_argument("input_csv", help="Path to input CSV (must have 'doi', 'abstract' columns)")
    parser.add_argument("output_csv", help="Path to output CSV (will be created)")
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model identifier (OpenRouter-style, e.g. openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--provider",
        default="default",
        choices=list(LLM_PROVIDER_SETTINGS.keys()),
        help="LLM provider (default: default)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key (optional; resolves from env if not provided)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM endpoint base URL (optional; resolves from env or provider default)",
    )
    args = parser.parse_args()

    # Launch the processing pipeline
    process_csv(
        args.input_csv,
        args.output_csv,
        args.model,
        args.provider,
        args.api_key,
        args.base_url,
    )


if __name__ == "__main__":
    main()
