"""Rechnet das Raster der Messreihen vor (E1 bis E5) und schreibt data/matrix_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse (rund eine Minute)
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht in data/matrix_grid.json)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probe nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kmx_constants as C  # noqa: E402
import kmx_evaluation as ev  # noqa: E402

JOBS = {"noise": ev.noise_cell, "spec": ev.spectrum_cell, "above": ev.above_cell, "err": ev.error_cell}


def _run(task):
    kind, args = task
    return kind, JOBS[kind](*args)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = [("noise", (p, n)) for p in sorted(C.NOISE_PS, reverse=True) for n in C.NOISE_NS]                 # die teuren Zellen zuerst
    tasks += [("spec", (q,)) for q in C.SPEC_QS]
    tasks += [("above", (t, n)) for t in C.TRUTHS for n in C.ABOVE_NS]
    tasks += [("err", (t, C.ERR_P, n)) for t in C.TRUTHS for n in C.ERR_NS]
    tasks += [("err", ("kette", C.WIDE_P, n)) for n in C.WIDE_NS]
    if quick:
        tasks = [("noise", (10, 30)), ("spec", (0.5,)), ("above", ("faktor", 120)), ("err", ("kette", 50, 60))]
    t0 = time.time()
    out = {k: [] for k in JOBS}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for kind, row in pool.map(_run, tasks, chunksize=1):
            out[kind].append(row)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("matrix_grid_quick.json" if quick else "matrix_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
