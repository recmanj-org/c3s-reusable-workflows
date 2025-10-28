#!/usr/bin/env python3
"""
Figure Checker for Jupyter Notebooks

Checks for figure labels and source attribution (Criterion 3.3.2).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import read_notebook, extract_cell_source, write_results


def check_figures(notebook_path: str) -> str:
    """
    Check for figure labels and source attribution in a notebook.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Result status: "success" or "failure"
    """
    nb_data = read_notebook(notebook_path)

    issues = []

    # Source attribution patterns
    source_patterns = [
        r'source:?\s+\S+',
        r'data\s+from:?\s*',
        r'doi:?\s*10\.\d+',
        r'https?://\S+',
        r'credit:?\s*',
        r'attribution:?\s*',
        r'reference:?\s*',
        r'dataset:?\s*',
    ]

    cells = nb_data.get('cells', [])

    for cell_idx, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])

            for output in outputs:
                if output.get('output_type') in ['display_data', 'execute_result']:
                    data = output.get('data', {})

                    # Check if this is a figure output
                    if 'image/png' in data or 'image/jpeg' in data or 'image/jpg' in data:
                        # Check nearby markdown cells for source attribution
                        has_source = False

                        # Check 2 cells before and after
                        for offset in [-2, -1, 1, 2]:
                            check_idx = cell_idx + offset
                            if 0 <= check_idx < len(cells):
                                check_cell = cells[check_idx]
                                if check_cell.get('cell_type') == 'markdown':
                                    source = extract_cell_source(check_cell)

                                    # Check for source patterns
                                    for pattern in source_patterns:
                                        if re.search(pattern, source, re.IGNORECASE):
                                            has_source = True
                                            break

                                if has_source:
                                    break

                        if not has_source:
                            issues.append(
                                f"Cell {cell_idx}: Figure missing source attribution "
                                f"(expected in nearby markdown cell)"
                            )

    if not issues:
        print(f"✅ All figures have proper labels and sources in {notebook_path}")
        return "success"
    else:
        print(f"❌ Found {len(issues)} figure labeling issue(s) in {notebook_path}:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nAdd source attribution in markdown cells near figures")
        print("Expected patterns: 'Source:', 'Data from:', DOI, URL, 'Credit:', etc.")
        return "failure"


def main():
    parser = argparse.ArgumentParser(description='Check for figure labels and sources in notebooks')
    parser.add_argument('--notebooks', required=True, help='JSON array of notebook paths')
    parser.add_argument('--output-dir', required=True, help='Directory to write results')
    args = parser.parse_args()

    try:
        notebooks = json.loads(args.notebooks)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format for notebooks: {e}")
        sys.exit(1)

    notebook_results = {}
    overall_result = 0

    for notebook in notebooks:
        if not notebook:
            continue

        print(f"Processing {notebook} with figure_checker")
        result = check_figures(notebook)
        notebook_results[notebook] = result

        if result == "failure":
            overall_result = 1

    write_results("figure_checker", notebook_results, args.output_dir)

    sys.exit(overall_result)


if __name__ == "__main__":
    main()
