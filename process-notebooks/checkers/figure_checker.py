#!/usr/bin/env python3
"""
Figure Checker for Jupyter Notebooks

Checks for figure labels, axis labels, and source attribution (Criteria 3.3.2 and 3.4.1).
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
from utils import read_notebook, extract_cell_source, write_results


PLOTTING_FUNCTIONS = {
    'plot', 'scatter', 'bar', 'barh', 'hist', 'boxplot', 'violinplot',
    'imshow', 'contour', 'contourf', 'pcolormesh', 'errorbar', 'fill',
    'fill_between', 'stackplot', 'stem', 'step', 'loglog', 'semilogx',
    'semilogy', 'pie', 'hexbin',
}

LABEL_FUNCTIONS = {'xlabel', 'ylabel', 'title', 'set_xlabel', 'set_ylabel', 'set_title'}


class MatplotlibCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.plotting_calls = []
        self.label_calls = []

    def visit_Call(self, node):
        func_name = self._get_func_name(node.func)
        if func_name in PLOTTING_FUNCTIONS:
            self.plotting_calls.append({'function': func_name, 'line': node.lineno})
        if func_name in LABEL_FUNCTIONS:
            args = self._extract_args(node)
            self.label_calls.append({'function': func_name, 'line': node.lineno, 'args': args})
        self.generic_visit(node)

    def _get_func_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _extract_args(self, node):
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Constant):
                args.append(arg.value)
            elif isinstance(arg, ast.Str):
                args.append(arg.s)
        return args


def analyze_cell_ast(cell_source: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(cell_source)
        visitor = MatplotlibCallVisitor()
        visitor.visit(tree)

        xlabel_empty = ylabel_empty = False
        for label_call in visitor.label_calls:
            func = label_call['function']
            args = label_call['args']
            if args and isinstance(args[0], str):
                if 'xlabel' in func and not args[0].strip():
                    xlabel_empty = True
                if 'ylabel' in func and not args[0].strip():
                    ylabel_empty = True

        return {
            'has_plot': len(visitor.plotting_calls) > 0,
            'has_xlabel': any('xlabel' in c['function'] for c in visitor.label_calls),
            'has_ylabel': any('ylabel' in c['function'] for c in visitor.label_calls),
            'xlabel_empty': xlabel_empty,
            'ylabel_empty': ylabel_empty,
        }
    except SyntaxError:
        return analyze_cell_regex(cell_source)


def analyze_cell_regex(cell_source: str) -> Dict[str, Any]:
    plot_re = re.compile(r'\b(?:plt|ax\w*)\.(plot|scatter|bar|hist|imshow)\s*\(')
    xlabel_re = re.compile(r'\b(?:plt|ax\w*)\.(?:set_)?xlabel\s*\(\s*["\'](.+?)["\']\s*\)')
    ylabel_re = re.compile(r'\b(?:plt|ax\w*)\.(?:set_)?ylabel\s*\(\s*["\'](.+?)["\']\s*\)')

    xlabel_match = xlabel_re.search(cell_source)
    ylabel_match = ylabel_re.search(cell_source)

    return {
        'has_plot': bool(plot_re.search(cell_source)),
        'has_xlabel': bool(xlabel_match),
        'has_ylabel': bool(ylabel_match),
        'xlabel_empty': xlabel_match and not xlabel_match.group(1).strip() if xlabel_match else False,
        'ylabel_empty': ylabel_match and not ylabel_match.group(1).strip() if ylabel_match else False,
    }


def has_exemption(cell_source: str) -> bool:
    return bool(re.search(r'#\s*no-label-check', cell_source))


def check_figures(notebook_path: str, check_axis_labels: bool = True,
                  require_xlabel: bool = True, require_ylabel: bool = True,
                  check_window: int = 2) -> str:
    """
    Check for figure labels, axis labels, and source attribution in a notebook.

    Args:
        notebook_path: Path to the notebook file
        check_axis_labels: Whether to check for axis labels
        require_xlabel: Require x-axis labels
        require_ylabel: Require y-axis labels
        check_window: Number of cells to check before/after for labels

    Returns:
        Result status: "success" or "failure"
    """
    nb_data = read_notebook(notebook_path)
    cells = nb_data.get('cells', [])
    issues = []

    source_patterns = [
        r'source:?\s+\S+', r'data\s+from:?\s*', r'doi:?\s*10\.\d+',
        r'https?://\S+', r'credit:?\s*', r'attribution:?\s*',
        r'reference:?\s*', r'dataset:?\s*',
    ]

    cell_analyses = []
    for cell in cells:
        if cell.get('cell_type') == 'code':
            source = extract_cell_source(cell)
            if has_exemption(source):
                cell_analyses.append({'has_plot': False, 'exempted': True, 'has_image': False})
            else:
                analysis = analyze_cell_ast(source)
                analysis['has_image'] = any(
                    key.startswith('image/') for output in cell.get('outputs', [])
                    for key in output.get('data', {}).keys()
                )
                analysis['exempted'] = False
                cell_analyses.append(analysis)
        else:
            cell_analyses.append({'has_plot': False, 'exempted': False, 'has_image': False})

    for cell_idx, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])

            for output in outputs:
                if output.get('output_type') in ['display_data', 'execute_result']:
                    data = output.get('data', {})

                    if 'image/png' in data or 'image/jpeg' in data or 'image/jpg' in data:
                        has_source = False
                        for offset in [-2, -1, 1, 2]:
                            check_idx = cell_idx + offset
                            if 0 <= check_idx < len(cells):
                                check_cell = cells[check_idx]
                                if check_cell.get('cell_type') == 'markdown':
                                    source = extract_cell_source(check_cell)
                                    for pattern in source_patterns:
                                        if re.search(pattern, source, re.IGNORECASE):
                                            has_source = True
                                            break
                                if has_source:
                                    break

                        if not has_source:
                            issues.append(
                                f"Cell {cell_idx + 1}: Figure missing source attribution "
                                f"(expected in nearby markdown cell)"
                            )

                        if check_axis_labels:
                            analysis = cell_analyses[cell_idx]
                            if analysis.get('has_plot', False) and not analysis.get('exempted', False):
                                start_idx = max(0, cell_idx - check_window)
                                end_idx = min(len(cell_analyses), cell_idx + check_window + 1)

                                has_xlabel = has_ylabel = False
                                for j in range(start_idx, end_idx):
                                    cell_data = cell_analyses[j]
                                    if cell_data.get('has_xlabel', False) and not cell_data.get('xlabel_empty', False):
                                        has_xlabel = True
                                    if cell_data.get('has_ylabel', False) and not cell_data.get('ylabel_empty', False):
                                        has_ylabel = True

                                missing_labels = []
                                if require_xlabel and not has_xlabel:
                                    missing_labels.append('x-axis label')
                                if require_ylabel and not has_ylabel:
                                    missing_labels.append('y-axis label')

                                if missing_labels:
                                    missing_str = ', '.join(missing_labels)
                                    issues.append(f"Cell {cell_idx + 1}: Figure missing {missing_str}")

    if not issues:
        print(f"✅ All figures have proper labels and sources in {notebook_path}")
        return "success"
    else:
        print(f"❌ Found {len(issues)} figure issue(s) in {notebook_path}:")
        for issue in issues:
            print(f"   - {issue}")
        if check_axis_labels:
            print("\nFor axis labels, use: plt.xlabel('...') / plt.ylabel('...')")
            print("Or: ax.set_xlabel('...') / ax.set_ylabel('...')")
            print("To skip validation: # no-label-check")
        print("\nFor source attribution, add to nearby markdown cells")
        print("Expected patterns: 'Source:', 'Data from:', DOI, URL, 'Credit:', etc.")
        return "failure"


def main():
    """Main entry point for figure checker."""
    parser = argparse.ArgumentParser(description='Check for figure labels, axis labels, and sources in notebooks')
    parser.add_argument('--notebooks', required=True, help='JSON array of notebook paths')
    parser.add_argument('--output-dir', required=True, help='Directory to write results')
    parser.add_argument('--check-axis-labels', default='true', help='Check for axis labels (default: true)')
    parser.add_argument('--require-xlabel', default='true', help='Require x-axis labels (default: true)')
    parser.add_argument('--require-ylabel', default='true', help='Require y-axis labels (default: true)')
    parser.add_argument('--check-window', type=int, default=2, help='Cells to check before/after (default: 2)')
    args = parser.parse_args()

    try:
        notebooks = json.loads(args.notebooks)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format for notebooks: {e}")
        sys.exit(1)

    check_axis_labels = args.check_axis_labels.lower() in ('true', '1', 'yes')
    require_xlabel = args.require_xlabel.lower() in ('true', '1', 'yes')
    require_ylabel = args.require_ylabel.lower() in ('true', '1', 'yes')

    notebook_results = {}
    overall_result = 0

    for notebook in notebooks:
        if not notebook:
            continue

        print(f"Processing {notebook} with figure_checker")
        result = check_figures(
            notebook,
            check_axis_labels=check_axis_labels,
            require_xlabel=require_xlabel,
            require_ylabel=require_ylabel,
            check_window=args.check_window,
        )
        notebook_results[notebook] = result

        if result == "failure":
            overall_result = 1

    write_results("figure_checker", notebook_results, args.output_dir)

    sys.exit(overall_result)


if __name__ == "__main__":
    main()
