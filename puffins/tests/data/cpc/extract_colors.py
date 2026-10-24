"""Write the one-decimal values and colors of a CPC ONI or RONI table as CSV.

Usage: ``python extract_colors.py PAGE.html OUT.csv``, where PAGE.html is a
saved copy of one of the two pages named in this folder's README.  Each table
cell becomes a row ``year,season,shown,phase``: the value as the table shows
it, and 1 for a cell colored El Niño (red), -1 for La Niña (blue) and 0 for
an uncolored one.  The RONI page marks the colors with the classes
``roni-warm``, ``roni-cold`` and ``roni-normal``, the ONI page with the styles
``color:red``, ``color:blue`` and ``color:black``.
"""

import csv
import re
import sys
from pathlib import Path

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]  # fmt: skip
PHASES = {
    "roni-warm": 1, "roni-cold": -1, "roni-normal": 0,
    "color:red": 1, "color:blue": -1, "color:black": 0,
}  # fmt: skip
# A marker, then any tags nested inside the cell, then its value.
CELL = re.compile(
    r'(?:class="(roni-\w+)"|style="(color:\w+)")[^>]*>(?:<[^>]+>)*(-?\d+\.\d)<'
)


def extract(html: str) -> list[tuple[int, str, str, int]]:
    """The table's cells in time order, as (year, season, shown, phase)."""
    table = html[html.index("<table") : html.rindex("</table>")]
    cells = []
    # Split at each row's start rather than its end, since the RONI page
    # leaves its last row unclosed.
    for row in table.split("<tr")[1:]:
        year = re.search(r">(\d{4})<", row)
        if year is None:
            continue
        found = CELL.findall(row)
        for season, (cls, style, shown) in zip(SEASONS, found, strict=False):
            cells.append((int(year.group(1)), season, shown, PHASES[cls or style]))
        if len(found) > len(SEASONS):
            raise ValueError(f"{len(found)} cells in the row for {year.group(1)}")
    return cells


def main(page: Path, out: Path) -> None:
    cells = extract(page.read_text(encoding="utf-8", errors="replace"))
    years = sorted({year for year, *_ in cells})
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError("a year is missing from the table")
    if any(len([c for c in cells if c[0] == y]) != 12 for y in years[:-1]):
        raise ValueError("a year before the last has fewer than 12 seasons")
    with out.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["year", "season", "shown", "phase"])
        writer.writerows(cells)
    first, last = cells[0], cells[-1]
    print(f"{len(cells)} seasons, {first[1]} {first[0]} to {last[1]} {last[0]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
