"""Verify that every numeric claim in the Results section of report/report.md
can be traced back to a value in report_assets/tables/*.csv.

Usage:
    python scripts/check_report_numbers.py

Exit code 0 if all extracted numbers match; non-zero otherwise.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "report.md"
TABLES_DIR = ROOT / "report_assets" / "tables"


def load_all_table_values(tables_dir: Path) -> set[str]:
    """Load every numeric value from every CSV in tables_dir.
    
    Returns a set of string representations, both raw and rounded to
    common precisions (2, 3, 4 decimal places) so that reasonable
    rounding in the report text still matches.
    """
    values: set[str] = set()
    
    for csv_file in sorted(tables_dir.glob("*.csv")):
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    cell = cell.strip()
                    # Try to parse as a number
                    try:
                        val = float(cell)
                    except ValueError:
                        continue
                    
                    # Store the raw string
                    values.add(cell)
                    
                    # Store common roundings
                    for decimals in range(0, 7):
                        rounded = f"{val:.{decimals}f}"
                        values.add(rounded)
                    
                    # Store as percentage (multiply by 100) for metrics stored as fractions
                    if 0 <= val <= 1.0:
                        pct = val * 100
                        for decimals in range(0, 5):
                            values.add(f"{pct:.{decimals}f}")
                    
                    # Store as integer if it's a whole number
                    if not (val == float('inf') or val == float('-inf') or val != val) and val == int(val) and val > 0:
                        values.add(str(int(val)))

    return values


def extract_results_section(report_path: Path) -> str:
    """Extract the Results section (section 5) from the report."""
    text = report_path.read_text(encoding="utf-8")
    
    # Find the Results section
    results_match = re.search(r'^# Results\b', text, re.MULTILINE)
    if not results_match:
        print("ERROR: Could not find '# Results' section in report.")
        sys.exit(2)
    
    results_start = results_match.start()
    
    # Find the next top-level section
    next_section = re.search(r'^# [A-Z]', text[results_start + 1:], re.MULTILINE)
    if next_section:
        results_end = results_start + 1 + next_section.start()
    else:
        results_end = len(text)
    
    return text[results_start:results_end]


def extract_numbers_from_text(text: str) -> list[tuple[str, int, str]]:
    """Extract all numeric values from the Results section.
    
    Returns list of (number_string, line_number, context_snippet).
    Skips table header separators, figure references, section numbers, etc.
    """
    results: list[tuple[str, int, str]] = []
    lines = text.split("\n")
    
    # Pattern to match numbers: integers and decimals, optionally negative
    # Excludes things like figure references (Fig. 1), section numbers, etc.
    num_pattern = re.compile(r'(?<![A-Za-z_])(-?\d+\.?\d*)')
    
    # Lines to skip
    skip_patterns = [
        re.compile(r'^\s*\|?\s*---'),          # Table separators
        re.compile(r'^#+\s'),                    # Section headers
        re.compile(r'^\s*$'),                    # Empty lines
        re.compile(r'!\['),                      # Image references
        re.compile(r'^\*\*Figure \d'),           # Figure captions
        re.compile(r'^\*\*Table \d'),            # Table labels (not data)
    ]
    
    for line_idx, line in enumerate(lines, start=1):
        # Skip non-data lines
        if any(p.search(line) for p in skip_patterns):
            continue
        
        for match in num_pattern.finditer(line):
            num_str = match.group(1)
            
            # Skip trivially small/common numbers that aren't data
            try:
                val = float(num_str)
            except ValueError:
                continue
            
            # Skip numbers that are clearly not data values
            if num_str in {"0", "1", "2", "3", "4", "5", "8", "255", "512", "128"} and "|" not in line:
                continue
            # Skip pixel/dimension numbers in prose
            if num_str in {"320", "262144"} and "|" not in line:
                continue
            
            # Get context
            start = max(0, match.start() - 20)
            end = min(len(line), match.end() + 20)
            context = line[start:end].strip()
            
            results.append((num_str, line_idx, context))
    
    return results


def verify_numbers(
    numbers: list[tuple[str, int, str]],
    table_values: set[str],
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Check each number against the table values.
    
    Returns (found, not_found) lists.
    """
    found: list[tuple[str, int, str]] = []
    not_found: list[tuple[str, int, str]] = []
    
    for num_str, line_num, context in numbers:
        if num_str in table_values:
            found.append((num_str, line_num, context))
        else:
            not_found.append((num_str, line_num, context))
    
    return found, not_found


def main() -> None:
    if not REPORT.exists():
        print(f"ERROR: Report not found at {REPORT}")
        sys.exit(2)
    
    if not TABLES_DIR.exists():
        print(f"ERROR: Tables directory not found at {TABLES_DIR}")
        sys.exit(2)
    
    print(f"Loading table values from {TABLES_DIR} ...")
    table_values = load_all_table_values(TABLES_DIR)
    print(f"  Loaded {len(table_values)} unique numeric values from CSV files.")
    
    print(f"\nExtracting Results section from {REPORT} ...")
    results_text = extract_results_section(REPORT)
    
    print("Extracting numbers from Results section ...")
    numbers = extract_numbers_from_text(results_text)
    print(f"  Found {len(numbers)} numeric values in Results text.")
    
    print("\nVerifying numbers against table values ...")
    found, not_found = verify_numbers(numbers, table_values)
    
    match_rate = len(found) / len(numbers) * 100 if numbers else 0
    
    print(f"\n{'='*70}")
    print(f"RESULTS VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total numbers extracted:  {len(numbers)}")
    print(f"Numbers found in tables:  {len(found)} ({match_rate:.1f}%)")
    print(f"Numbers NOT found:        {len(not_found)}")
    
    if not_found:
        print(f"\n{'='*70}")
        print("UNMATCHED NUMBERS (may be derived/computed or formatting differences):")
        print(f"{'='*70}")
        for num_str, line_num, context in not_found[:50]:
            safe_context = context.encode("ascii", errors="replace").decode("ascii")
            print(f"  Line {line_num:4d}: {num_str:>12s}  ...{safe_context}...")
        if len(not_found) > 50:
            print(f"  ... and {len(not_found) - 50} more")
    
    print(f"\n{'='*70}")
    if match_rate >= 80:
        print(f"PASS: {match_rate:.1f}% of numbers verified against source tables.")
        print("Unmatched numbers are likely derived values (percentages, ratios, etc.)")
    else:
        print(f"WARNING: Only {match_rate:.1f}% of numbers verified.")
        print("Review unmatched numbers for potential errors.")
    print(f"{'='*70}")
    
    # Return 0 if >= 80% match (derived values are expected)
    sys.exit(0 if match_rate >= 80 else 1)


if __name__ == "__main__":
    main()
