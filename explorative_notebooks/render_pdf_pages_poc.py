from pathlib import Path
import fitz
import argparse


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300,
):
    """
    Render every PDF page as a high-resolution PNG.
    """

    doc = fitz.open(pdf_path)

    for page_idx in range(len(doc)):

        page = doc[page_idx]

        pix = page.get_pixmap(
            dpi=dpi,
            alpha=False,
        )

        output_file = (
            output_dir
            / f"{pdf_path.stem}_page_{page_idx + 1:03d}.png"
        )

        pix.save(output_file)

    doc.close()


def main():

    parser = argparse.ArgumentParser(
        description="Render all PDF pages to PNG"
    )

    parser.add_argument(
        "pdf",
        help="Path to PDF"
    )

    parser.add_argument(
        "--output-dir",
        default="page_render_output",
        help="Output directory"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    render_pdf_pages(
        pdf_path=pdf_path,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    print(
        f"Rendered all pages from {pdf_path.name}"
    )
    print(
        f"Output: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()