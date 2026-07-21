"""Entry point: ``python -m abel.validation`` launches the validation GUI."""

from __future__ import annotations

import sys

from abel.validation.gui import launch_validation_gui


def main() -> int:
    return launch_validation_gui()


if __name__ == "__main__":
    sys.exit(main())
