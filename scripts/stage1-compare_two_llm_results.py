import json
from pathlib import Path

import pandas as pd


# ============================================================
# User settings
# ============================================================

csv1_path = "/Users/florianscheiber/PycharmProjects/llm-climate-neutrality-review/outputs_llm_runs/abstracts_output_20260529_openai_gpt4o-mini.csv"
csv2_path = "/Users/florianscheiber/PycharmProjects/llm-climate-neutrality-review/outputs_llm_runs/output_20260610_validation_gpt-4o-mini.csv"

output_json = "/Users/florianscheiber/PycharmProjects/llm-climate-neutrality-review/compare_two_llm_results/compare_gpt4o-mini_20260529_and_validation_20260610/grading_comparison_summary.json"
output_differences_csv = "/Users/florianscheiber/PycharmProjects/llm-climate-neutrality-review/compare_two_llm_results/compare_gpt4o-mini_20260529_and_validation_20260610/grading_comparison_differences.csv"

doi_col = "doi"

question_cols = [
    "q1_climate_neutrality",
    "q2_region",
    "q3_sector",
    "q4_method",
]

allowed_values = {"yes", "no", "unclear"}
missing_label = "missing"


# ============================================================
# Helper functions
# ============================================================

def read_csv_checked(path: str, name: str) -> pd.DataFrame:
    """
    Read CSV and check that all required columns exist.
    """
    df = pd.read_csv(path, sep=",")

    required_cols = [doi_col] + question_cols
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{name}: Missing required columns: {missing_cols}"
        )

    return df[required_cols].copy()

def drop_missing_dois(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list]:
    """
    Drop rows with missing or empty DOI values.
    These rows cannot be compared because DOI is the paper identifier.
    """
    missing_mask = df[doi_col].isna() | (df[doi_col].astype(str).str.strip() == "")

    dropped_rows = []

    for idx in df.index[missing_mask]:
        dropped_rows.append({
            "csv": name,
            "row_index": int(idx),
        })

    df_clean = df.loc[~missing_mask].copy()

    return df_clean, dropped_rows

def check_duplicate_dois(df: pd.DataFrame, name: str) -> list:
    """
    Return duplicate DOIs. Empty list means no duplicates.
    """
    duplicated = df[df[doi_col].duplicated(keep=False)][doi_col]
    return sorted(duplicated.dropna().astype(str).unique().tolist())


def collect_invalid_values(df: pd.DataFrame, name: str) -> list:
    """
    Find values in q1-q4 that are neither yes/no/unclear nor missing.
    Missing values are reported separately and are not invalid.
    """
    invalid_entries = []

    for col in question_cols:
        non_missing = df[col].dropna()

        invalid_mask = ~non_missing.isin(allowed_values)
        invalid_values = non_missing[invalid_mask]

        for idx, value in invalid_values.items():
            invalid_entries.append({
                "csv": name,
                "row_index": int(idx),
                "doi": df.loc[idx, doi_col],
                "question": col,
                "value": value,
            })

    return invalid_entries


def collect_missing_values(df: pd.DataFrame, name: str) -> list:
    """
    Find missing values in q1-q4.
    """
    missing_entries = []

    for col in question_cols:
        missing_mask = df[col].isna()

        for idx in df.index[missing_mask]:
            missing_entries.append({
                "csv": name,
                "row_index": int(idx),
                "doi": df.loc[idx, doi_col],
                "question": col,
            })

    return missing_entries


def pct(part: int, whole: int) -> float:
    """
    Safe percentage helper.
    """
    if whole == 0:
        return 0.0
    return round(100.0 * part / whole, 3)


def value_for_comparison(value):
    """
    Convert NaN to explicit missing label for comparison.
    """
    if pd.isna(value):
        return missing_label
    return value


# ============================================================
# Main analysis
# ============================================================

def compare_gradings(csv1_path: str, csv2_path: str) -> dict:
    df1 = read_csv_checked(csv1_path, "csv1")
    df2 = read_csv_checked(csv2_path, "csv2")

    # --------------------------------------------------------
    # Drop rows with missing DOI
    # --------------------------------------------------------

    df1, dropped_missing_dois_csv1 = drop_missing_dois(df1, "csv1")
    df2, dropped_missing_dois_csv2 = drop_missing_dois(df2, "csv2")
    # --------------------------------------------------------
    # Hard checks: duplicate DOI
    # --------------------------------------------------------

    duplicate_dois_csv1 = check_duplicate_dois(df1, "csv1")
    duplicate_dois_csv2 = check_duplicate_dois(df2, "csv2")

    if duplicate_dois_csv1 or duplicate_dois_csv2:
        raise ValueError(
            "Duplicate DOI detected. Aborting.\n"
            f"csv1 duplicates: {duplicate_dois_csv1}\n"
            f"csv2 duplicates: {duplicate_dois_csv2}"
        )

    # --------------------------------------------------------
    # Hard checks: invalid values
    # --------------------------------------------------------

    invalid_values_csv1 = collect_invalid_values(df1, "csv1")
    invalid_values_csv2 = collect_invalid_values(df2, "csv2")

    if invalid_values_csv1 or invalid_values_csv2:
        raise ValueError(
            "Invalid grading values detected. Aborting.\n"
            f"Allowed values are: {sorted(allowed_values)}\n"
            f"Invalid values in csv1: {invalid_values_csv1}\n"
            f"Invalid values in csv2: {invalid_values_csv2}"
        )

    # --------------------------------------------------------
    # Missing values: report, but do not abort
    # --------------------------------------------------------

    missing_values_csv1 = collect_missing_values(df1, "csv1")
    missing_values_csv2 = collect_missing_values(df2, "csv2")

    # --------------------------------------------------------
    # DOI overlap
    # --------------------------------------------------------

    dois1 = set(df1[doi_col])
    dois2 = set(df2[doi_col])

    dois_in_both = sorted(dois1 & dois2)
    only_csv1 = sorted(dois1 - dois2)
    only_csv2 = sorted(dois2 - dois1)

    doi_overlap = {
        "n_csv1": len(dois1),
        "n_csv2": len(dois2),
        "n_in_both": len(dois_in_both),
        "n_only_csv1": len(only_csv1),
        "n_only_csv2": len(only_csv2),
        "only_csv1": only_csv1,
        "only_csv2": only_csv2,
    }

    # --------------------------------------------------------
    # Restrict to common DOIs
    # --------------------------------------------------------

    df1_common = (
        df1[df1[doi_col].isin(dois_in_both)]
        .set_index(doi_col)
        .sort_index()
    )

    df2_common = (
        df2[df2[doi_col].isin(dois_in_both)]
        .set_index(doi_col)
        .sort_index()
    )

    # --------------------------------------------------------
    # Cell-level, paper-level, question-level comparisons
    # --------------------------------------------------------

    total_cells = 0
    equal_cells = 0
    different_cells = 0

    question_level = {}
    transition_counts_overall = {}
    transition_counts_by_question = {}

    different_papers = []
    differences_rows = []

    for q in question_cols:
        question_equal = 0
        question_different = 0
        transition_counts_by_question[q] = {}

        for doi in dois_in_both:
            v1 = value_for_comparison(df1_common.loc[doi, q])
            v2 = value_for_comparison(df2_common.loc[doi, q])

            total_cells += 1

            if v1 == v2:
                equal_cells += 1
                question_equal += 1
            else:
                different_cells += 1
                question_different += 1

                transition = f"{v1} -> {v2}"

                transition_counts_overall[transition] = (
                    transition_counts_overall.get(transition, 0) + 1
                )

                transition_counts_by_question[q][transition] = (
                    transition_counts_by_question[q].get(transition, 0) + 1
                )

                differences_rows.append({
                    "doi": doi,
                    "question": q,
                    "csv1_value": v1,
                    "csv2_value": v2,
                    "transition": transition,
                })

        question_level[q] = {
            "n_equal": question_equal,
            "n_different": question_different,
            "pct_equal": pct(question_equal, len(dois_in_both)),
            "pct_different": pct(question_different, len(dois_in_both)),
        }

    # --------------------------------------------------------
    # Paper-level differences
    # --------------------------------------------------------

    for doi in dois_in_both:
        paper_differences = {}

        for q in question_cols:
            v1 = value_for_comparison(df1_common.loc[doi, q])
            v2 = value_for_comparison(df2_common.loc[doi, q])

            if v1 != v2:
                paper_differences[q] = {
                    "csv1": v1,
                    "csv2": v2,
                }

        if paper_differences:
            different_papers.append({
                "doi": doi,
                "differences": paper_differences,
            })

    n_common_papers = len(dois_in_both)
    n_papers_with_any_difference = len(different_papers)
    n_papers_identical_all_questions = (
        n_common_papers - n_papers_with_any_difference
    )

    # --------------------------------------------------------
    # Summary object
    # --------------------------------------------------------

    result = {
        "input_files": {
            "csv1": str(csv1_path),
            "csv2": str(csv2_path),
        },
        "settings": {
            "doi_normalization": "none",
            "csv_separator": ",",
            "allowed_values": sorted(allowed_values),
            "missing_label": missing_label,
            "missing_values_count_as_differences": True,
            "questions": question_cols,
        },
        "doi_overlap": doi_overlap,
        "overall_cell_comparison": {
            "total_cells_compared": total_cells,
            "n_equal": equal_cells,
            "n_different": different_cells,
            "pct_equal": pct(equal_cells, total_cells),
            "pct_different": pct(different_cells, total_cells),
        },
        "paper_level_comparison": {
            "n_common_papers": n_common_papers,
            "n_papers_identical_all_questions": n_papers_identical_all_questions,
            "n_papers_with_any_difference": n_papers_with_any_difference,
            "pct_papers_identical_all_questions": pct(
                n_papers_identical_all_questions,
                n_common_papers,
            ),
            "pct_papers_with_any_difference": pct(
                n_papers_with_any_difference,
                n_common_papers,
            ),
        },
        "question_level_comparison": question_level,
        "difference_transitions_overall": dict(
            sorted(transition_counts_overall.items())
        ),
        "difference_transitions_by_question": {
            q: dict(sorted(transitions.items()))
            for q, transitions in transition_counts_by_question.items()
        },
        "different_papers": different_papers,
        "data_quality": {
            "dropped_missing_dois_csv1": dropped_missing_dois_csv1,
            "dropped_missing_dois_csv2": dropped_missing_dois_csv2,
            "n_dropped_missing_dois_csv1": len(dropped_missing_dois_csv1),
            "n_dropped_missing_dois_csv2": len(dropped_missing_dois_csv2),

            "duplicate_dois_csv1": duplicate_dois_csv1,
            "duplicate_dois_csv2": duplicate_dois_csv2,
            "invalid_values_csv1": invalid_values_csv1,
            "invalid_values_csv2": invalid_values_csv2,
            "missing_values_csv1": missing_values_csv1,
            "missing_values_csv2": missing_values_csv2,
            "n_missing_values_csv1": len(missing_values_csv1),
            "n_missing_values_csv2": len(missing_values_csv2),
        },
        "_differences_rows": differences_rows,
    }

    return result


# ============================================================
# Run and save
# ============================================================

if __name__ == "__main__":
    result = compare_gradings(csv1_path, csv2_path)

    # Save full JSON summary.
    result_for_json = {
        key: value
        for key, value in result.items()
        if key != "_differences_rows"
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_for_json, f, indent=2, ensure_ascii=False)

    # Save differences as CSV for manual inspection.
    differences_df = pd.DataFrame(result["_differences_rows"])

    if len(differences_df) > 0:
        differences_df.to_csv(output_differences_csv, index=False)
    else:
        # Create an empty but well-structured CSV.
        differences_df = pd.DataFrame(
            columns=[
                "doi",
                "question",
                "csv1_value",
                "csv2_value",
                "transition",
            ]
        )
        differences_df.to_csv(output_differences_csv, index=False)

    # Console summary.
    print("Comparison finished.")
    print()
    print("DOI overlap:")
    print(json.dumps(result_for_json["doi_overlap"], indent=2))
    print()
    print("Overall cell comparison:")
    print(json.dumps(result_for_json["overall_cell_comparison"], indent=2))
    print()
    print("Paper-level comparison:")
    print(json.dumps(result_for_json["paper_level_comparison"], indent=2))
    print()
    print("Question-level comparison:")
    print(json.dumps(result_for_json["question_level_comparison"], indent=2))
    print()
    print(f"Saved JSON summary to: {output_json}")
    print(f"Saved differences CSV to: {output_differences_csv}")