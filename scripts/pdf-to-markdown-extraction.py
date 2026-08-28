from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Iterable

import pymupdf
import pymupdf4llm


DEFAULT_INPUT_DIR = Path(
    "inputs/fulltext_pdf/included-after-stage-3-100"
)
DEFAULT_OUTPUT_DIR = Path(
    "intermediate/pdf-extraction"
)
# DPI used when rasterising vector-graphic figure regions to PNG.
DEFAULT_IMAGE_DPI = 200
# Skip images whose width or height is below this fraction of the page
# (filters out logos, rules, icons and other decoration).
DEFAULT_IMAGE_SIZE_LIMIT = 0.05


def extract_paper_id(pdf_path: Path) -> str:
    match = re.match(r"^(\d{3})", pdf_path.stem)
    if not match:
        raise RuntimeError(
            f"Could not extract 3-digit paper id from {pdf_path.name}"
        )
    return match.group(1)


def extract_paper_kind(pdf_path: Path) -> str:
    stem = pdf_path.stem.lower()
    if stem.endswith("-main"):
        return "main"
    if re.search(r"-(supplement|supplment)-[1-9]$", stem):
        return stem.rsplit("-", 1)[-1]
    raise RuntimeError(
        f"Unsupported pdf name {pdf_path.name}; expected -main or -supplement-n"
    )


def iter_pdf_paths(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.glob("*.pdf"))


def _relativise_image_paths(page_markdown: str, images_dir: Path) -> str:
    """pymupdf4llm writes image links prefixed with the full ``image_path``.
    Rewrite them to be relative to the markdown file (``images/<name>.png``).
    The library always emits forward slashes in these links."""
    prefix = images_dir.as_posix().rstrip("/") + "/"
    return page_markdown.replace(f"]({prefix}", "](images/")


def extract_page_markdown(
    doc: pymupdf.Document,
    pdf_stem: str,
    images_dir: Path,
    dpi: int,
) -> list[str]:
    """Return one Markdown string per page.

    Tables are reconstructed as GitHub-flavoured Markdown tables and the
    figures/images (embedded bitmaps and vector-graphic figure regions)
    are written to ``images_dir`` and linked inline 
    """
    chunks = pymupdf4llm.to_markdown(
        doc,
        filename=pdf_stem,
        page_chunks=True,
        write_images=True,
        image_path=str(images_dir),
        image_format="png",
        image_size_limit=DEFAULT_IMAGE_SIZE_LIMIT,
        dpi=dpi,
        show_progress=False,
    )
    return [
        _relativise_image_paths(str(chunk.get("text", "")).strip(), images_dir)
        for chunk in chunks
    ]


def extract_pdf_to_markdown_and_images(
    pdf_path: Path,
    output_root: Path,
    dpi: int = DEFAULT_IMAGE_DPI,
) -> Path:
    paper_id = extract_paper_id(pdf_path)
    paper_kind = extract_paper_kind(pdf_path)

    paper_output_dir = output_root / paper_id
    images_dir = paper_output_dir / "images"
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = paper_output_dir / f"{pdf_path.stem}.md"

    parts: list[str] = []
    parts.append(f"# {pdf_path.name}")
    parts.append("")
    parts.append(f"- paper_id: {paper_id}")
    parts.append(f"- paper_kind: {paper_kind}")
    parts.append(f"- source_pdf: {pdf_path.name}")
    parts.append("")

    with pymupdf.open(pdf_path) as doc:
        page_count = len(doc)
        page_markdown = extract_page_markdown(
            doc,
            pdf_stem=pdf_path.stem,
            images_dir=images_dir,
            dpi=dpi,
        )

    for page_index in range(page_count):
        page_number = page_index + 1

        parts.append(f"## Page {page_number}")
        parts.append("")

        page_text = (
            page_markdown[page_index]
            if page_index < len(page_markdown)
            else ""
        )
        if page_text:
            parts.append(page_text)
        else:
            parts.append("_No extractable text found on this page._")
        parts.append("")

    markdown_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    return markdown_path


def process_input_folder(input_dir: Path, output_root: Path, dpi: int) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for pdf_path in iter_pdf_paths(input_dir):
        print(f"Processing {pdf_path.name}..." )
        written_files.append(
            extract_pdf_to_markdown_and_images(
                pdf_path=pdf_path,
                output_root=output_root,
                dpi=dpi,
            )
        )

    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PDFs to markdown plus extracted figure images"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing the input PDFs",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder to write extracted markdown and images",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_IMAGE_DPI,
        help="DPI used to rasterise vector-graphic figure regions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    written_files = process_input_folder(
        input_dir=input_dir,
        output_root=output_dir,
        dpi=args.dpi,
    )

    print(f"Exported {len(written_files)} PDFs to {output_dir}")


if __name__ == "__main__":
    main()
