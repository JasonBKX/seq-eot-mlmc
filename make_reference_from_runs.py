"""Build reference.json from completed runs of a (large-N) study, e.g. for darcy2d.

    python make_reference_from_runs.py RUN_DIR CONFIG METHOD OUT_DIR

value = mean of that method's estimates for that configuration, se = sd / sqrt(repeats).
The target hash is computed exactly as the driver does, so --reference-file accepts the file
only for the same problem, data and observable. The reference is E_{pi_L}[phi] up to the
estimator's own bias (O(1/N) for MLSMC) plus the reported SE; use many more particles than the
studies it will score.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import beskos_transport_mh as tm

run_dir, config, method, out_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
meta = json.loads((run_dir / "metadata.json").read_text())
cfg = meta["config"]
problem = tm.make_problem(cfg, meta["target"]["y"])
target = tm.target_description(problem)
records = [json.loads(p.read_text()) for p in sorted((run_dir / "records").glob(f"{config}_*_{method}.json"))]
if len(records) < 2:
    raise SystemExit(f"need >= 2 records for {config}/{method}, found {len(records)}")
est = np.array([r["estimate"] for r in records])
info = {"target": target, "target_hash": tm.canonical_hash(target), "mode": "runs",
        "value": float(est.mean()), "se": float(est.std(ddof=1) / math.sqrt(len(est))),
        "quadrature_change": None, "source": {"run_dir": str(run_dir), "configuration": config,
        "method": method, "repeats": len(est), "N": records[0]["N"], "estimates": est.tolist()},
        "note": "Mean of independent large-N runs; SE from their spread."}
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "reference.json").write_text(json.dumps(info, indent=1))
print(f"reference {info['value']:.10g} +- {info['se']:.3g} from {len(est)} runs -> {out_dir / 'reference.json'}")
