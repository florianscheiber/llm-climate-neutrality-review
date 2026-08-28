# scripts/03-stage3c-merge-extractions.py  (stage 3c, step 3)
#
# Merge the per-paper mitigation / costs / emissions CSVs written by
# 01-stage3c-data-extraction.py (and 02-stage3c-candidates-extraction.py,
# via --extra-input-dir) into three combined CSVs.
#
# For every row:
#
# - add `paper_id` (the 3-digit output folder name, without leading zeros)
# - join `Title`, `Publication_Year` and `DOI` from the full-text screening
#   table, matched on the article id (that table stores ids without leading
#   zeros)
#
# INPUT:
#
#     outputs/output-stage3c/
#     └── 001/
#         ├── 001-mitigation.csv
#         ├── 001-costs.csv
#         └── 001-emissions.csv
#
#     inputs/included_after_full_text_screening.xlsx
#
# OUTPUT:
#
#     <output_dir>/merged-mitigation.csv
#     <output_dir>/merged-costs.csv
#     <output_dir>/merged-emissions.csv
#
# USAGE IN TERMINAL:  (run from the repo root)
#
#     python3 scripts/03-stage3c-merge-extractions.py
#
#     python3 scripts/03-stage3c-merge-extractions.py \
#         outputs/output-stage3c \
#         outputs/output-stage3c \
#         --screening inputs/included_after_full_text_screening.xlsx \
#         --extra-input-dir outputs/output-stage3c-candidates

from pathlib import Path
import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET


CATEGORIES = ("mitigation", "costs", "emissions")

# Columns prepended to every merged row.
JOIN_COLUMNS = ["paper_id", "Title", "Publication_Year", "DOI", "extraction_pass"]

DEFAULT_INPUT_DIR = Path("outputs/output-stage3c")
DEFAULT_SCREENING = Path("inputs/included_after_full_text_screening.xlsx")

XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# ============================================================================
# MINIMAL XLSX READING
# ============================================================================

def _column_letters(cell_ref: str) -> str:
    return re.match(r"[A-Za-z]+", cell_ref).group(0).upper()


def read_xlsx_rows(xlsx_path: Path):
    """Yield rows of the first worksheet as {column_letter: text}.

    Handles shared strings, inline strings and plain values -- which is all
    the screening table contains. Avoids a pandas / openpyxl dependency.
    """
    with zipfile.ZipFile(xlsx_path) as archive:

        shared_strings = []

        if "xl/sharedStrings.xml" in archive.namelist():
            sst = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for string_item in sst:
                shared_strings.append(
                    "".join(
                        node.text or ""
                        for node in string_item.iter(f"{XLSX_NS}t")
                    )
                )

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

        for row in sheet.iter(f"{XLSX_NS}row"):

            values = {}

            for cell in row.findall(f"{XLSX_NS}c"):

                cell_ref = cell.attrib.get("r", "")
                if not cell_ref:
                    continue

                cell_type = cell.attrib.get("t")

                if cell_type == "inlineStr":
                    inline = cell.find(f"{XLSX_NS}is")
                    text = (
                        "".join(
                            node.text or ""
                            for node in inline.iter(f"{XLSX_NS}t")
                        )
                        if inline is not None
                        else ""
                    )
                else:
                    value_node = cell.find(f"{XLSX_NS}v")
                    if value_node is None:
                        text = ""
                    elif cell_type == "s":
                        text = shared_strings[int(value_node.text)]
                    else:
                        text = value_node.text or ""

                values[_column_letters(cell_ref)] = text.strip()

            yield values


def load_paper_metadata(xlsx_path: Path) -> dict:
    """Return {article_id_int: {"Title": str, "Publication_Year": str, "DOI": str}}."""

    rows = list(read_xlsx_rows(xlsx_path))

    header_row = None
    for row in rows:
        if "Article ID" in row.values():
            header_row = row
            break

    if header_row is None:
        raise RuntimeError(f"'Article ID' header not found in {xlsx_path}")

    def find_column(label: str) -> str:
        for letter, value in header_row.items():
            if value == label:
                return letter
        raise RuntimeError(f"column '{label}' not found in {xlsx_path}")

    id_col = find_column("Article ID")
    title_col = find_column("Title")
    year_col = find_column("publication year")
    doi_col = find_column("DOI")

    metadata = {}
    seen_header = False

    for row in rows:

        if row is header_row:
            seen_header = True
            continue

        if not seen_header:
            continue

        raw_id = row.get(id_col, "").strip()
        if not re.fullmatch(r"\d+", raw_id):
            continue

        metadata[int(raw_id)] = {
            "Title": row.get(title_col, "").strip(),
            "Publication_Year": row.get(year_col, "").strip(),
            "DOI": row.get(doi_col, "").strip(),
        }

    return metadata


# ============================================================================
# MERGING
# ============================================================================

def iter_paper_dirs(input_dir: Path):
    return sorted(
        entry
        for entry in input_dir.iterdir()
        if entry.is_dir() and re.fullmatch(r"\d+", entry.name)
    )


def merge_category(category: str, input_dirs, metadata: dict):
    """Return (fieldnames, rows, papers_without_metadata) for one category.

    ``input_dirs`` is a list of (directory, pass_label) pairs; rows from
    every directory are concatenated and tagged with their pass_label in
    the ``extraction_pass`` column.
    """

    fieldnames = list(JOIN_COLUMNS)
    merged_rows = []
    missing_metadata = []

    for input_dir, pass_label in input_dirs:

        for paper_dir in iter_paper_dirs(input_dir):

            folder_id = paper_dir.name
            paper_id = str(int(folder_id))

            csv_path = paper_dir / f"{folder_id}-{category}.csv"
            if not csv_path.exists():
                continue

            meta = metadata.get(int(folder_id))
            if meta is None:
                missing_metadata.append(paper_id)
                meta = {"Title": "", "Publication_Year": "", "DOI": ""}

            with open(csv_path, "r", encoding="utf-8", newline="") as f:

                reader = csv.DictReader(f, delimiter=";")

                for column in reader.fieldnames or []:
                    if column and column not in fieldnames:
                        fieldnames.append(column)

                for row in reader:
                    out_row = {
                        "paper_id": paper_id,
                        "Title": meta["Title"],
                        "Publication_Year": meta["Publication_Year"],
                        "DOI": meta["DOI"],
                        "extraction_pass": pass_label,
                    }
                    out_row.update(row)
                    merged_rows.append(out_row)

    return fieldnames, merged_rows, missing_metadata


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Merge stage3c per-paper mitigation/costs/emissions CSVs and "
            "join Title + DOI from the full-text screening table"
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help="stage3c output folder (default: %(default)s)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="where to write merged-*.csv (default: same as input_dir)",
    )
    parser.add_argument(
        "--screening",
        default=str(DEFAULT_SCREENING),
        help="full-text screening .xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "--extra-input-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="additional per-paper CSV folder to fold into the merge, e.g. "
             "the stage3c-candidates second pass. Repeatable. Rows from it "
             "are tagged extraction_pass=<folder name>.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    screening_path = Path(args.screening)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if not screening_path.is_file():
        raise FileNotFoundError(f"screening file not found: {screening_path}")

    input_dirs = [(input_dir, "stage3c")]
    for extra in args.extra_input_dir:
        extra_path = Path(extra)
        if not extra_path.is_dir():
            raise FileNotFoundError(f"extra input directory not found: {extra_path}")
        input_dirs.append((extra_path, extra_path.name))

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_paper_metadata(screening_path)
    print(
        f"Loaded Title/Publication_Year/DOI for {len(metadata)} papers "
        f"from {screening_path.name}"
    )

    for category in CATEGORIES:

        fieldnames, rows, missing = merge_category(
            category, input_dirs, metadata
        )

        out_path = output_dir / f"merged-{category}.csv"

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=";",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"{category}: {len(rows)} rows -> {out_path}")

        if missing:
            print(
                f"  WARNING: no screening Title/DOI for paper id(s): "
                f"{', '.join(sorted(set(missing), key=int))}"
            )


if __name__ == "__main__":
    main()
