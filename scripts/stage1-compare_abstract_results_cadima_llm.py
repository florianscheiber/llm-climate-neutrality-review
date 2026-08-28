import csv
import json
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

# HIER ANPASSEN
LLM_RUN_OUTPUT = "llm_cadima_api_claude-4.6-sonnet"
CADIMA_STAGE = "cadima_after_fulltext" # cadima_after_abstract or cadima_after_fulltext


LLM_RESULTS_CSV = f"outputs_llm_runs_abstract_screening/{LLM_RUN_OUTPUT}.csv"
CADIMA_INCLUDED_CSV = f"cadima_outputs/included_{CADIMA_STAGE}.csv"
CADIMA_EXCLUDED_CSV = f"cadima_outputs/excluded_{CADIMA_STAGE}.csv"

# Output-Verzeichnis (gleich wie Input)
OUTPUT_DIR = Path(".")

# ============================================================================
# HELPER: CSV mit automatischer Trennzeichen-Erkennung laden
# ============================================================================

def _choose_delimiter(sample: str):
    """
    Wähle das wahrscheinlichste Trennzeichen basierend auf Häufigkeit im Sample.
    Priorität: , ; \t |
    """
    counts = {
        ",": sample.count(","),
        ";": sample.count(";"),
        "\t": sample.count("\t"),
        "|": sample.count("|"),
    }
    # Sortiere nach Häufigkeit, nehme das mit der höchsten Anzahl (Fallback: ',')
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return best[0] if best[1] > 0 else ","

def load_csv_auto_delimiter(filepath, doi_column):
    """
    Lädt CSV mit automatischer Trennzeichen-Erkennung.
    Rückgabe: (data_dict, skipped_rows)
      - data_dict: {doi (lowercase) -> row_dict}
      - skipped_rows: list von dicts: {"line_no": int, "raw": str, "row": dict_or_none}
    Robust gegenüber:
      - unterschiedlichen Trennzeichen (heuristische Erkennung + Fallback)
      - zusammengeführtem Header (z.B. "a,b,c" als einzige Feldname)
      - unterschiedlichen DOI-Spaltennamen (case-insensitive)
      - fehlenden/NULL DOI-Werten (überspringen, sammeln)
    """
    data = {}
    skipped = []  # list of {"line_no":..., "raw":..., "row":...}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            sample = f.read(8192)
            f.seek(0)

            # Heuristische Wahl des Delimiters
            guessed = None
            try:
                # Versuche zuerst den csv.Sniffer (wenn er nicht komplett versagt)
                dialect = csv.Sniffer().sniff(sample)
                guessed = dialect.delimiter
            except Exception:
                guessed = _choose_delimiter(sample)

            # Erster Versuch mit guessed delimiter
            f.seek(0)
            reader = csv.DictReader(f, delimiter=guessed)
            fieldnames = reader.fieldnames or []

            # Falls DictReader nur eine Feldname liefert, aber diese Feldname enthält
            # Kommas (z.B. "doi,q1_climate..."), dann nehme explizit ',' als delimiter
            if len(fieldnames) == 1 and "," in fieldnames[0] and guessed != ",":
                print(f"⚠️ Warnung: Header sieht zusammengeführt aus in {filepath}. Versuch mit Komma-Delimiter.")
                f.seek(0)
                reader = csv.DictReader(f, delimiter=",")
                fieldnames = reader.fieldnames or []

            # Ermittele DOI-Spaltenname case-insensitive
            doi_key = None
            target_lower = doi_column.strip().lower()
            for fn in fieldnames:
                if fn and fn.strip().lower() == target_lower:
                    doi_key = fn
                    break
            if doi_key is None:
                for fn in fieldnames:
                    if fn and fn.strip().lower() == "doi":
                        doi_key = fn
                        break

            if doi_key is None:
                print(f"⚠️ DOI-Spalte '{doi_column}' nicht gefunden in {filepath}. Verfügbare Spalten: {fieldnames}")
                # Wir verwenden trotzdem den gewünschten Namen, aber das führt zu vielen leeren Werten
                doi_key = doi_column

            # Iteriere und sammle fehlende DOI-Zeilen mit Zeilennummern
            total = 0
            header_lines = 1  # DictReader behandelt erste Zeile als header
            for i, row in enumerate(reader, start=1):
                total += 1
                # Versuche Rohzeile zu rekonstruieren (für Troubleshooting)
                # csv.DictReader gibt row, wir können die Rohwerte per join zeigen
                raw_preview = None
                try:
                    # Zeilenindex relativ zur Datei (angenommen header ist Zeile 1)
                    line_no = header_lines + i
                    # Build a printable raw preview from the row values
                    raw_preview = "|".join([str(v) for v in row.values()]) if row else ""
                except Exception:
                    line_no = header_lines + i
                    raw_preview = ""

                raw_val = row.get(doi_key) if row else None
                if raw_val is None:
                    skipped.append({"line_no": line_no, "raw": raw_preview, "row": row})
                    continue
                doi = str(raw_val).strip().lower()
                if doi:
                    data[doi] = row
                else:
                    skipped.append({"line_no": line_no, "raw": raw_preview, "row": row})

            if skipped:
                print(f"⚠️ {len(skipped)} von {total} Zeilen in {filepath} ohne DOI (oder leere DOI).")
    except Exception as e:
        print(f"❌ Fehler beim Laden von {filepath}: {e}")
        raise

    return data, skipped

# ============================================================================
# PREPROCESSING
# ============================================================================

def load_cadima_data():
    """
    Lädt CADIMA-CSVs (included + excluded).
    Checkt: keine DOI darf in beiden CSVs sein.
    Gibt dict zurück: {doi -> "included" oder "excluded"}
    """
    print("📖 Lade CADIMA-Daten...")

    included, included_skipped = load_csv_auto_delimiter(CADIMA_INCLUDED_CSV, "DOI")
    excluded, excluded_skipped = load_csv_auto_delimiter(CADIMA_EXCLUDED_CSV, "DOI")

    # Wenn viele Zeilen ohne DOI, gebe ein paar Beispiele aus zum Troubleshooting
    def _print_skipped_examples(skipped_list, name):
        if not skipped_list:
            return
        print(f"\n--- Beispiele für fehlende DOI in {name} (max 50) ---")
        for entry in skipped_list[:50]:
            print(f"Zeile {entry['line_no']}: {entry['raw']}")
        print("--- Ende Beispiele ---\n")

    if included_skipped:
        _print_skipped_examples(included_skipped, CADIMA_INCLUDED_CSV)
    if excluded_skipped:
        _print_skipped_examples(excluded_skipped, CADIMA_EXCLUDED_CSV)

    # Check: Überschneidung?
    overlap = set(included.keys()) & set(excluded.keys())
    if overlap:
        raise ValueError(
            f"❌ FEHLER: {len(overlap)} DOI(s) in BEIDEN CADIMA-CSVs! "
            f"Beispiel: {list(overlap)[:3]}"
        )

    # Kombiniertes dict
    cadima_data = {}
    for doi in included.keys():
        cadima_data[doi] = "included"
    for doi in excluded.keys():
        cadima_data[doi] = "excluded"

    print(f"✅ CADIMA geladen: {len(included)} included, {len(excluded)} excluded")
    return cadima_data

def load_llm_data():
    """
    Lädt LLM-Ergebnisse.
    Klassifiziert: "included" wenn q1–q4 ALLE "yes", sonst "excluded".
    Gibt dict zurück: {doi -> "included" oder "excluded"}
    """
    print("📖 Lade LLM-Daten...")

    llm_raw, llm_skipped = load_csv_auto_delimiter(LLM_RESULTS_CSV, "doi")
    if llm_skipped:
        # Wenn Header falsch erkannt wurde, load_csv_auto_delimiter bereits versucht Fallback.
        # Hier nur Hinweis geben, falls viele Zeilen ohne DOI gefunden wurden.
        print(f"⚠️ Hinweis: {len(llm_skipped)} Zeilen ohne DOI in {LLM_RESULTS_CSV} gefunden (werden übersprungen).")
        # Gib ein paar Beispiele aus
        for entry in llm_skipped[:10]:
            print(f"  LLM fehlende DOI - Zeile {entry['line_no']}: {entry['raw']}")

    llm_data = {}

    for doi, row in llm_raw.items():
        # Schütze vor None-Werten in einzelnen Spalten
        q1 = (row.get("q1_climate_neutrality") or "").strip().lower()
        q2 = (row.get("q2_region") or "").strip().lower()
        q3 = (row.get("q3_sector") or "").strip().lower()
        q4 = (row.get("q4_method") or "").strip().lower()

        answers = [q1, q2, q3, q4]

        if "no" in answers:
            llm_data[doi] = "excluded"
        else:
            llm_data[doi] = "included"

    print(f"✅ LLM geladen: {len(llm_data)} papers")
    return llm_data

# ============================================================================
# VERGLEICH: Overlap & nur in einem vorhanden
# ============================================================================

def compare_datasets(cadima_data, llm_data):
    """
    Vergleicht CADIMA und LLM.
    Gibt 3 Sets zurück:
      - both: DOIs in beiden
      - only_llm: DOIs nur in LLM
      - only_cadima: DOIs nur in CADIMA
    """
    print("\n🔍 Vergleiche Datensätze...")

    cadima_dois = set(cadima_data.keys())
    llm_dois = set(llm_data.keys())

    both = cadima_dois & llm_dois
    only_llm = llm_dois - cadima_dois
    only_cadima = cadima_dois - llm_dois

    print(f"  📊 In BEIDEN: {len(both)}")
    print(f"  📊 Nur in LLM: {len(only_llm)}")
    print(f"  📊 Nur in CADIMA: {len(only_cadima)}")

    return both, only_llm, only_cadima

# ============================================================================
# KATEGORISIERUNG: Nur Papers in BEIDEN nach Relevanz
# ============================================================================

def categorize_overlapping(cadima_data, llm_data, both_dois):
    """
    Für Papers, die in BEIDEN vorhanden sind:
    - both_relevant: beide "included"
    - both_not_relevant: beide "excluded"
    - only_llm_relevant: nur LLM "included", CADIMA "excluded"
    - only_cadima_relevant: nur CADIMA "included", LLM "excluded"
    """
    print("\n📋 Kategorisiere überlappende Papers...")

    both_relevant = []
    both_not_relevant = []
    only_llm_relevant = []
    only_cadima_relevant = []

    for doi in both_dois:
        cadima_label = cadima_data[doi]
        llm_label = llm_data[doi]

        if cadima_label == "included" and llm_label == "included":
            both_relevant.append(doi)
        elif cadima_label == "excluded" and llm_label == "excluded":
            both_not_relevant.append(doi)
        elif llm_label == "included" and cadima_label == "excluded":
            only_llm_relevant.append(doi)
        elif cadima_label == "included" and llm_label == "excluded":
            only_cadima_relevant.append(doi)

    print(f"  ✅ Beide relevant: {len(both_relevant)}")
    print(f"  ❌ Beide nicht relevant: {len(both_not_relevant)}")
    print(f"  ⚠️  Nur LLM relevant: {len(only_llm_relevant)}")
    print(f"  ⚠️  Nur CADIMA relevant: {len(only_cadima_relevant)}")

    return {
        "both_relevant": both_relevant,
        "both_not_relevant": both_not_relevant,
        "only_llm_relevant": only_llm_relevant,
        "only_cadima_relevant": only_cadima_relevant,
    }

# ============================================================================
# SPEICHERN: JSON-Output
# ============================================================================

def save_json(filename, data):
    """Speichert dict als JSON."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Gespeichert: {filepath}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("VERGLEICH: CADIMA vs. LLM Abstract Screening")
    print("=" * 70)

    try:
        # Laden
        cadima_data = load_cadima_data()
        llm_data = load_llm_data()

        # Vergleich: Overlap
        both, only_llm, only_cadima = compare_datasets(cadima_data, llm_data)

        # Speichere Overlap-Info
        overlap_info = {
            "total_in_both": len(both),
            "total_only_llm": len(only_llm),
            "total_only_cadima": len(only_cadima),
            "dois_in_both": sorted(list(both)),
            "dois_only_llm": sorted(list(only_llm)),
            "dois_only_cadima": sorted(list(only_cadima)),
        }
        save_json(f"analyses_llm_vs_cadima/{CADIMA_STAGE}/{LLM_RUN_OUTPUT}/overlap_info.json", overlap_info)

        # Kategorisierung (nur overlapping)
        categories = categorize_overlapping(cadima_data, llm_data, both)

        # Speichere Kategorien
        categories_info = {
            "both_relevant": sorted(categories["both_relevant"]),
            "both_not_relevant": sorted(categories["both_not_relevant"]),
            "only_llm_relevant": sorted(categories["only_llm_relevant"]),
            "only_cadima_relevant": sorted(categories["only_cadima_relevant"]),
            "summary": {
                "both_relevant_count": len(categories["both_relevant"]),
                "both_not_relevant_count": len(categories["both_not_relevant"]),
                "only_llm_relevant_count": len(categories["only_llm_relevant"]),
                "only_cadima_relevant_count": len(categories["only_cadima_relevant"]),
            }
        }
        save_json(f"analyses_llm_vs_cadima/{CADIMA_STAGE}/{LLM_RUN_OUTPUT}/comparison_categories.json", categories_info)

        print("\n" + "=" * 70)
        print("✅ FERTIG!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ FEHLER: {e}")
        raise

if __name__ == "__main__":
    main()

