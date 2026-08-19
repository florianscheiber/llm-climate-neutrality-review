# analyze_image_payload_sizes.py
#
# Estimates the PNG payload size that would be sent to OpenRouter
# after rendering all PDFs belonging to the same paper ID.
#
# Uses the same rendering settings as stage3-extraction.py.
#
# Usage:
#
# python3 analyze_image_payload_sizes.py papers
#

from pathlib import Path
import argparse
import re

import fitz
from PIL import Image
import io

PAGE_RENDER_DPI = 250
max_dimension_value = 1800

OPENROUTER_LIMIT_MB = 30.0


def extract_article_id(pdf_path: Path):

    match = re.match(
        r"^(\d+)",
        pdf_path.stem
    )

    if not match:
        return None

    return match.group(1)


def render_page_to_png_bytes(page):

    pix = page.get_pixmap(
        dpi=PAGE_RENDER_DPI,
        alpha=False
    )

    img = Image.open(
        io.BytesIO(
            pix.tobytes("png")
        )
    )

    max_dimension = max(img.size)

    if max_dimension > max_dimension_value:

        scale = max_dimension_value / max_dimension

        new_size = (
            int(img.size[0] * scale),
            int(img.size[1] * scale),
        )

        img = img.resize(
            new_size,
            Image.LANCZOS,
        )

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="PNG"
    )

    return len(
        buffer.getvalue()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pdf_folder"
    )

    args = parser.parse_args()

    pdf_folder = Path(
        args.pdf_folder
    )

    groups = {}

    for pdf_path in pdf_folder.glob(
        "*.pdf"
    ):

        article_id = extract_article_id(
            pdf_path
        )

        if article_id is None:
            continue

        groups.setdefault(
            article_id,
            []
        ).append(pdf_path)

    print()
    print(
        f"Found {len(groups)} paper groups"
    )
    print()

    over_limit = []

    for article_id in sorted(
        groups.keys(),
        key=int
    ):

        pdfs = sorted(
            groups[article_id]
        )

        total_bytes = 0
        total_pages = 0

        for pdf_path in pdfs:

            doc = fitz.open(pdf_path)

            total_pages += len(doc)

            for page in doc:

                total_bytes += (
                    render_page_to_png_bytes(
                        page
                    )
                )

            doc.close()

        png_mb = (
            total_bytes
            / 1024
            / 1024
        )

        # Approximate base64 overhead (+33%)
        request_mb = png_mb * 1.33

        status = "OK"

        if request_mb > OPENROUTER_LIMIT_MB:

            status = "OVER_LIMIT"

            over_limit.append(
                (
                    article_id,
                    total_pages,
                    request_mb,
                )
            )

        print(
            f"{article_id:>5} | "
            f"{total_pages:>4} pages | "
            f"{request_mb:>6.1f} MB | "
            f"{status}"
        )

    print()
    print(
        f"Over limit: "
        f"{len(over_limit)} / {len(groups)} papers"
    )

    if over_limit:

        print()
        print(
            "Papers exceeding 30 MB:"
        )

        for (
            article_id,
            pages,
            size_mb,
        ) in over_limit:

            print(
                f"{article_id}: "
                f"{pages} pages, "
                f"{size_mb:.1f} MB"
            )


if __name__ == "__main__":
    main()