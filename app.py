#!/usr/bin/env python3
"""KPViz from a clone, without installing: `python app.py --help`.

The same as the `kpviz` command (kpviz/cli.py), except that ./data and
./sample_data are looked for next to this file.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from kpviz.cli import main
    sys.exit(main(home=Path(__file__).resolve().parent))
