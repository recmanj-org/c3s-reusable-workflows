#!/usr/bin/env python3
"""
Test Checker for Jupyter Notebooks

Checks for test files and runs coverage analysis.
Validates criteria 2.3.1 (test existence) and 2.3.2 (coverage threshold).
"""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import write_results


def check_tests(notebook_path: str, coverage_threshold: float = 80.0) -> str:
    """
    Check for test files and run coverage analysis.

    Args:
        notebook_path: Path to the notebook file
        coverage_threshold: Minimum coverage percentage required

    Returns:
        Result status: "success" or "failure"
    """
    # Check for test files (Criterion 2.3.1)
    test_patterns = ['test_*.py', '*_test.py', 'tests/**/*.py']
    test_files = []

    for pattern in test_patterns:
        test_files.extend(Path('.').glob(pattern))

    if not test_files:
        print(f"❌ No test files found for {notebook_path}")
        print("   Create test_*.py or tests/ directory with test files")
        return "failure"

    print(f"✅ Found {len(test_files)} test file(s)")
    for test_file in test_files:
        print(f"   - {test_file}")

    # Run tests with coverage (Criterion 2.3.2)
    print(f"\nRunning tests with coverage (threshold: {coverage_threshold}%)...")

    try:
        pytest_result = subprocess.run(
            ['pytest', '--cov=.', '--cov-report=xml', '--cov-report=term'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        print(pytest_result.stdout)
        if pytest_result.stderr:
            print(pytest_result.stderr)

        coverage_xml = Path('coverage.xml')
        if not coverage_xml.exists():
            print("❌ Coverage report not generated")
            return "failure"

        tree = ET.parse('coverage.xml')
        root = tree.getroot()
        coverage_pct = float(root.attrib.get('line-rate', 0)) * 100

        print(f"\nCode coverage: {coverage_pct:.1f}%")

        if coverage_pct < coverage_threshold:
            print(f"❌ Coverage {coverage_pct:.1f}% below threshold {coverage_threshold}%")
            return "failure"
        else:
            print(f"✅ Coverage {coverage_pct:.1f}% meets threshold {coverage_threshold}%")
            return "success"

    except subprocess.TimeoutExpired:
        print("❌ Tests timed out after 5 minutes")
        return "failure"
    except FileNotFoundError:
        print("❌ pytest not found. Install pytest and pytest-cov")
        return "failure"
    except Exception as e:
        print("❌ Error running tests: {e}")
        return "failure"


def main():
    parser = argparse.ArgumentParser(description='Check for tests and coverage in notebooks')
    parser.add_argument('--notebooks', required=True, help='JSON array of notebook paths')
    parser.add_argument('--output-dir', required=True, help='Directory to write results')
    parser.add_argument('--coverage-threshold', type=float, default=80.0,
                        help='Minimum coverage percentage (default: 80)')
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

        print(f"Processing {notebook} with test_checker")
        result = check_tests(notebook, args.coverage_threshold)
        notebook_results[notebook] = result

        if result == "failure":
            overall_result = 1

    write_results("test_checker", notebook_results, args.output_dir)

    sys.exit(overall_result)


if __name__ == "__main__":
    main()
