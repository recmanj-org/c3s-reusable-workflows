#!/usr/bin/env python3
"""
DOI Checker for Jupyter Notebooks

Checks if notebooks that use datasets with DOI metadata properly cite those DOIs (Criterion 1.2.5).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import read_notebook, extract_cell_source, write_results


def check_doi(notebook_path: str) -> str:
    """
    Check for DOI citations in a notebook.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Result status: "success", "failure", or "skipped"
    """
    # DOI regex patterns
    doi_patterns = [
        r'10\.\d{4,9}/[-._;()/:A-Z0-9]+',  # Standard DOI format
        r'doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)',  # doi.org URLs
        r'https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)'  # Full DOI URLs
    ]

    # Metadata fields that might contain DOI references
    metadata_fields = ['references', 'citation', 'doi', 'reference', 'Attributes']

    nb_data = read_notebook(notebook_path)

    # First, check if dataset metadata contains DOI references
    has_dataset_doi_metadata = False

    for cell in nb_data.get('cells', []):
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                # Check text outputs for dataset metadata with DOIs
                if 'text' in output:
                    text = output['text']
                    if isinstance(text, list):
                        text = ''.join(text)
                    # Check if this looks like xarray/dataset output with metadata
                    if any(field in text for field in metadata_fields):
                        # Check if metadata contains DOI patterns
                        for pattern in doi_patterns:
                            if re.search(pattern, text, re.IGNORECASE):
                                has_dataset_doi_metadata = True
                                break

                # Check data outputs for metadata
                data = output.get('data', {})
                for key, value in data.items():
                    if isinstance(value, (str, list)):
                        if isinstance(value, list):
                            value = ''.join(value)
                        if any(field in value for field in metadata_fields):
                            for pattern in doi_patterns:
                                if re.search(pattern, value, re.IGNORECASE):
                                    has_dataset_doi_metadata = True
                                    break

                if has_dataset_doi_metadata:
                    break

        if has_dataset_doi_metadata:
            break

    if not has_dataset_doi_metadata:
        print(f"INFO: No dataset DOI metadata found in {notebook_path}, skipping DOI check")
        return "skipped"

    found_dois = set()

    # Search in all cells for DOI citations
    for cell in nb_data.get('cells', []):
        # Check markdown cells
        if cell.get('cell_type') == 'markdown':
            source = extract_cell_source(cell)
            for pattern in doi_patterns:
                matches = re.findall(pattern, source, re.IGNORECASE)
                found_dois.update(matches)

        # Check code cell outputs
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                # Check text outputs
                if 'text' in output:
                    text = output['text']
                    if isinstance(text, list):
                        text = ''.join(text)
                    for pattern in doi_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        found_dois.update(matches)

                # Check data outputs (HTML, plain text, etc.)
                data = output.get('data', {})
                for key, value in data.items():
                    if isinstance(value, (str, list)):
                        if isinstance(value, list):
                            value = ''.join(value)
                        for pattern in doi_patterns:
                            matches = re.findall(pattern, value, re.IGNORECASE)
                            found_dois.update(matches)

    # Validate DOI format
    valid_doi_pattern = re.compile(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', re.IGNORECASE)
    valid_dois = [doi for doi in found_dois if valid_doi_pattern.match(doi)]

    if not valid_dois:
        print(f"ERROR: No valid DOI found in {notebook_path}")
        print("  Notebook uses dataset with DOI metadata but doesn't cite the DOI")
        print("  Please add DOI citation in markdown or ensure dataset metadata is visible")
        print("  Expected format: 10.xxxx/xxxxx")
        return "failure"
    else:
        print(f"Found {len(valid_dois)} valid DOI(s) in {notebook_path}:")
        for doi in sorted(valid_dois):
            print(f"  - {doi}")
        return "success"


def main():
    parser = argparse.ArgumentParser(description='Check for DOI citations in notebooks')
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

        print(f"Processing {notebook} with doi_checker")
        result = check_doi(notebook)
        notebook_results[notebook] = result

        if result == "failure":
            overall_result = 1

    write_results("doi_checker", notebook_results, args.output_dir)

    sys.exit(overall_result)


if __name__ == "__main__":
    main()
