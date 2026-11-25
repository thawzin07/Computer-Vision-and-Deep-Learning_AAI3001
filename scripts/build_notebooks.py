"""Build and validate Google Colab training notebooks (v2 dated by default)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from nb_letters import build_letters_notebook
from nb_letters_v2 import build_letters_v2_notebook
from nb_words import build_words_notebook
from nb_words_v2 import build_words_v2_notebook


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and schema-validate Colab notebooks.")
    parser.add_argument(
        "--include-v1",
        action="store_true",
        help="Also regenerate legacy v1 notebooks (disabled by default to preserve executed outputs).",
    )
    args = parser.parse_args()

    colab_dir = PROJECT_ROOT / "colab"
    words_v2 = build_words_v2_notebook(colab_dir / "01_Train_ASL_Words_Conformer_2026-11-19.ipynb")
    letters_v2 = build_letters_v2_notebook(colab_dir / "02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb")
    print(f"Built & validated: {words_v2}")
    print(f"Built & validated: {letters_v2}")

    if args.include_v1:
        words_nb = build_words_notebook(colab_dir / "01_Train_ASL_Words_Conformer.ipynb")
        letters_nb = build_letters_notebook(colab_dir / "02_Train_ASL_Letters_Hybrid.ipynb")
        print(f"Built & validated (v1): {words_nb}")
        print(f"Built & validated (v1): {letters_nb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
