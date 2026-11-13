"""Shared utilities for constructing and validating self-contained Colab notebooks."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_and_validate_notebook(
    cells: list[tuple[str, str]],
    out_path: Path,
) -> Path:
    """Create a Colab-ready .ipynb from (kind, source) pairs and validate syntax & schema."""
    nb_cells = []
    for idx, (kind, src) in enumerate(cells):
        text = src.strip("\n") + "\n"
        if kind == "md":
            nb_cells.append(new_markdown_cell(text))
        elif kind == "code":
            # Strip Colab shell magics (!pip / %) before AST syntax validation
            py_lines = [
                line if not line.lstrip().startswith(("!", "%")) else f"# magic: {line}"
                for line in text.splitlines()
            ]
            ast.parse("\n".join(py_lines), filename=f"{out_path.name}#cell_{idx}")
            nb_cells.append(new_code_cell(text))
        else:
            raise ValueError(f"Unknown cell kind: {kind}")

    nb = new_notebook(
        cells=nb_cells,
        metadata={
            "colab": {
                "provenance": [],
                "gpuType": "A100",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": "GPU",
        },
    )
    nbformat.validate(nb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    return out_path
