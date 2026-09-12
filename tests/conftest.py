"""Suite-wide test setup.

The suite runs in parallel (``-n 8 --dist worksteal`` in ``pyproject.toml``).  Each
xdist worker's BLAS / OpenMP / torch / numba pools would otherwise size themselves
to every core on the machine, so eight workers oversubscribe the CPU and the run
gets slower as workers are added.  Capping each worker's pools measured ~40 s vs
~46-54 s uncapped.  A serial run (``-n 0``) is left untouched.
"""

from __future__ import annotations

import os

_WORKER_THREADS = "4"

if os.environ.get("PYTEST_XDIST_WORKER"):
    # Read at import time by torch / numba / MKL, which the tests import lazily.
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMBA_NUM_THREADS"):
        os.environ.setdefault(_var, _WORKER_THREADS)
    # numpy's BLAS is already loaded by pytest plugins, so cap it at runtime too.
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(int(_WORKER_THREADS))
    except ImportError:
        pass
