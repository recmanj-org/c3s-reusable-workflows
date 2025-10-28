#!/usr/bin/env python3
"""
Metadata Checker for Jupyter Notebooks

Checks for version date information in notebooks or README files (Criterion 1.2.6).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import read_notebook, extract_cell_source, write_results


def check_metadata(notebook_path: str, check_readme: bool = True) -> str:
    """
    Check for version date metadata in a notebook.

    Args:
        notebook_path: Path to the notebook file
        check_readme: Whether to check README.md if not found in notebook

    Returns:
        Result status: "success" or "failure"
    """
    # Date patterns to search for
    date_patterns = [
        r'last\s+updated:?\s*(\d{4}-\d{2}-\d{2})',
        r'version:?\s*[\d.]+\s*\((\d{4}-\d{2}-\d{2})\)',
        r'modified:?\s*(\d{4}-\d{2}-\d{2})',
        r'date:?\s*(\d{4}-\d{2}-\d{2})',
        r'updated:?\s*(\d{4}-\d{2}-\d{2})',
    ]

    nb_data = read_notebook(notebook_path)

    found_date = None
    found_location = None

    # Check notebook metadata
    metadata = nb_data.get('metadata', {})
    for field in ['date', 'modified', 'version', 'last_updated']:
        if field in metadata:
            value = str(metadata[field])
            # Try to extract date from the metadata field
            for pattern in date_patterns:
                match = re.search(pattern, value, re.IGNORECASE)
                if match:
                    found_date = match.group(1)
                    found_location = f"metadata.{field}"
                    break
            if found_date:
                break

    # Check markdown cells if not found in metadata
    if not found_date:
        for cell_idx, cell in enumerate(nb_data.get('cells', [])):
            if cell.get('cell_type') == 'markdown':
                source = extract_cell_source(cell)

                # Search for date patterns
                for pattern in date_patterns:
                    match = re.search(pattern, source, re.IGNORECASE)
                    if match:
                        found_date = match.group(1)
                        found_location = f"markdown cell {cell_idx}"
                        break

                if found_date:
                    break

    # Check for README.md if not found in notebook
    if not found_date and check_readme:
        readme_path = Path(notebook_path).parent / "README.md"
        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()

                for pattern in date_patterns:
                    match = re.search(pattern, readme_content, re.IGNORECASE)
                    if match:
                        found_date = match.group(1)
                        found_location = "README.md"
                        break
            except Exception:
                pass

    if found_date:
        print(f"✅ Found version date for {notebook_path}: {found_date} (in {found_location})")
        return "success"
    else:
        print(f"❌ No version date found for {notebook_path}")
        print("   Add 'Last updated: YYYY-MM-DD' to notebook markdown or README.md")
        return "failure"


def main():
    parser = argparse.ArgumentParser(description='Check for version date metadata in notebooks')
    parser.add_argument('--notebooks', required=True, help='JSON array of notebook paths')
    parser.add_argument('--output-dir', required=True, help='Directory to write results')
    parser.add_argument('--check-readme', action='store_true', default=True,
                        help='Check README.md if not found in notebook')
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

        print(f"Processing {notebook} with metadata_checker")
        result = check_metadata(notebook, args.check_readme)
        notebook_results[notebook] = result

        if result == "failure":
            overall_result = 1

    write_results("metadata_checker", notebook_results, args.output_dir)

    sys.exit(overall_result)


if __name__ == "__main__":
    main()
