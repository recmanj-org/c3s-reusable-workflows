"""Shared utility functions for notebook checkers."""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def read_notebook(notebook_path: str) -> Dict[str, Any]:
    """
    Read and parse a Jupyter notebook file.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Parsed notebook data as dictionary

    Raises:
        SystemExit: If the notebook cannot be read
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading notebook {notebook_path}: {e}")
        sys.exit(1)


def extract_cell_source(cell: Dict[str, Any]) -> str:
    """
    Extract source code/markdown from a cell as a single string.

    Args:
        cell: Notebook cell dictionary

    Returns:
        Cell source as a single string
    """
    source = cell.get('source', [])
    if isinstance(source, list):
        return ''.join(source)
    return str(source)


def write_results(command_name: str, notebook_results: Dict[str, str], output_dir: str) -> None:
    """
    Write standardized JSON results file.

    Args:
        command_name: Name of the command/checker
        notebook_results: Dictionary mapping notebook paths to results (success/failure/skipped)
        output_dir: Directory to write the results file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_file = Path(output_dir) / f"{command_name}-results.json"

    results_data = {
        "command": command_name,
        "results": notebook_results
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
