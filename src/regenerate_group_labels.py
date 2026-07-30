"""Regenerate year-consistent group labels without reacquiring publications."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_csrankings_data import (
    SCALED_TARGET_AREAS,
    generate_groups_csv,
    load_generated_author_info,
)
from utils.config import GROUPS_CSV, PUBLICATIONS_CSV, YEAR_RANGE


def publication_authors(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        authors_column = next(
            (name for name in (reader.fieldnames or []) if name.strip().lower() == "authors"),
            None,
        )
        if authors_column is None:
            raise ValueError(f"Publications CSV has no Authors column: {path}")
        return {
            author.strip()
            for row in reader
            for author in str(row.get(authors_column) or "").split(";")
            if author.strip()
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild group labels for an existing publication dataset."
    )
    parser.add_argument("--publications-csv", type=Path, default=PUBLICATIONS_CSV)
    parser.add_argument(
        "--author-info-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "generated-author-info.csv",
    )
    parser.add_argument("--output-csv", type=Path, default=GROUPS_CSV)
    parser.add_argument("--year-start", type=int, default=YEAR_RANGE[0])
    parser.add_argument("--year-end", type=int, default=YEAR_RANGE[1])
    parser.add_argument("--target-areas", default=",".join(SCALED_TARGET_AREAS))
    args = parser.parse_args(argv)

    authors = publication_authors(args.publications_csv)
    author_rows = load_generated_author_info(args.author_info_csv)
    target_areas = [
        area.strip() for area in args.target_areas.split(",") if area.strip()
    ]
    generate_groups_csv(
        researchers=[(name, 0.0, {}) for name in sorted(authors)],
        author_rows=author_rows,
        target_areas=target_areas,
        enriched_authors=authors,
        output_path=args.output_csv,
        year_range=(args.year_start, args.year_end),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
