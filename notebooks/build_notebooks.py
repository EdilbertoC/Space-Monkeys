# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Build .ipynb notebooks from the percent-format .py sources in this folder.

The .py file is the source of truth (easy to diff and review); run
`python notebooks/build_notebooks.py` after editing it.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
CELL = re.compile(r"^# %%( \[markdown\])?\s*$", re.MULTILINE)


def to_cells(source: str) -> list[dict]:
    cells = []
    marks = list(CELL.finditer(source))
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(source)
        body = source[mark.end() : end].strip("\n")
        if mark.group(1):
            lines = [re.sub(r"^# ?", "", line) for line in body.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(lines)})
        else:
            lines = [re.sub(r"^# (%pip .*)$", r"\1", line) for line in body.splitlines()]
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": _lines(lines),
                }
            )
    return cells


def _lines(lines: list[str]) -> list[str]:
    return [line + "\n" for line in lines[:-1]] + lines[-1:] if lines else []


def build(py_path: Path) -> Path:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": to_cells(py_path.read_text(encoding="utf-8")),
    }
    out = py_path.with_suffix(".ipynb")
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    targets = [Path(p) for p in sys.argv[1:]] or sorted(HERE.glob("*.py"))
    for path in targets:
        if path.name == Path(__file__).name:
            continue
        print("Built", build(path))
