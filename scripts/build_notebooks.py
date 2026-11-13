"""Build and validate both self-contained Google Colab training notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from nb_letters import build_letters_notebook
from nb_words import build_words_notebook


def main() -> int:
    colab_dir = PROJECT_ROOT / "colab"
    words_nb = build_words_notebook(colab_dir / "01_Train_ASL_Words_Conformer.ipynb")
    letters_nb = build_letters_notebook(colab_dir / "02_Train_ASL_Letters_Hybrid.ipynb")
    print(f"Built & validated: {words_nb}")
    print(f"Built & validated: {letters_nb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
