#!/usr/bin/env python3
"""
Accessibility Checker for Jupyter Notebooks

Checks for alt-text on all images (Criterion 3.1.3).
"""

import argparse
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from utils import read_notebook, extract_cell_source, write_results


def check_accessibility(notebook_path: str) -> str:
    """
    Check for alt-text on all images in a notebook.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Result status: "success" or "failure"
    """
    nb_data = read_notebook(notebook_path)

    issues = []

    for cell_idx, cell in enumerate(nb_data.get('cells', [])):
        # Check markdown cells for images
        if cell.get('cell_type') == 'markdown':
            source = extract_cell_source(cell)

            # Check HTML img tags
            soup = BeautifulSoup(source, 'html.parser')
            for img in soup.find_all('img'):
                alt = img.get('alt', '').strip()
                if not alt:
                    issues.append(f"Cell {cell_idx}: <img> tag missing alt text")

            # Check markdown image syntax: ![alt](url)
            md_images = re.findall(r'!\[(.*?)\]\([^)]*\)', source)
            ref_images = re.findall(r'!\[(.*?)\]\[[^\]]*\]', source)
            for idx, alt_text in enumerate(md_images + ref_images):
                if not alt_text.strip():
                    issues.append(f"Cell {cell_idx}: Markdown image missing alt text")

        # Check output cells for figures
        elif cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            for output in outputs:
                if output.get('output_type') in ['display_data', 'execute_result']:
                    # Check for image outputs
                    data = output.get('data', {})
                    if 'image/png' in data or 'image/jpeg' in data or 'image/jpg' in data:
                        metadata = output.get('metadata', {})
                        # Check for alt text in metadata
                        if not metadata.get('alt_text') and not metadata.get('alt'):
                            issues.append(f"Cell {cell_idx}: Figure output missing alt text metadata")

    if not issues:
        print(f"✅ All images have alt-text in {notebook_path}")
        return "success"
    else:
        print(f"❌ Found {len(issues)} accessibility issue(s) in {notebook_path}:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nAdd alt-text to all images for screen reader compatibility")
        return "failure"


def main():
    parser = argparse.ArgumentParser(description='Check for image alt-text in notebooks')
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

        print(f"Processing {notebook} with accessibility_checker")
        result = check_accessibility(notebook)
        notebook_results[notebook] = result

        if result == "failure":
            overall_result = 1

    write_results("accessibility_checker", notebook_results, args.output_dir)

    sys.exit(overall_result)


if __name__ == "__main__":
    main()
