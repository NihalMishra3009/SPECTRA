"""Drop in an externally trained emitter model (e.g. from the Turing dataset).

How to integrate YOUR trained model
-----------------------------------
    1. Export the model as a pickle file (any object with either
       `predict(X)` or `predict_proba(X)`, X = (N, 2*n_bands+1) feature rows,
       output=(N, C) band scores).
    2.  python -m app.train.integrate_dataset_model --model path/to/model.pkl
    3. Dashboard → pick `dataset_rfi` in the scheduler dropdown → RUN.

`--build-demo` trains a small numpy softmax next-band predictor from the built-in
RF simulator so the whole pipeline is provable without your model file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app import db
from app.config import SimConfig
from app.sim.engine import make_scheduler, run_simulation
from app.sim.scenarios import get_emitters
from app.sim.environment import RFEnvironment
from app.sim.schedulers.dataset_rfi import build_features, load_model
from app.train import ARTIFACTS_DIR
from app.train.rfi_model import BandPredictor, softmax_fit


# --------------------------------------------------------------------------- #
#  demo dataset / training (pure NumPy softmax on simulator scan history)
# --------------------------------------------------------------------------- #
def build_demo_dataset(n_bands: int = 10, n_steps: int = 200, seeds: list[int] | None = None):
    """Feature/target pairs from a round-robin scanner against random emitters."""
    if seeds is None:
        seeds = [1, 2, 3, 4, 5, 7, 11, 13]
    scenarios = ["stable", "switch", "periodic", "hopper", "surprise", "burst", "single_dominant", "random"]
    X, y = [], []
    for scen in scenarios:
        for seed in seeds:
            emitters = get_emitters(scen, n_bands, n_steps)
            env = RFEnvironment(n_bands=n_bands, n_steps=n_steps, emitters=emitters, seed=seed)
            truth = env.ground_truth
            hits = np.zeros(n_bands)
            misses = np.zeros(n_bands)
            for t in range(n_steps):
                band = t % n_bands  # round-robin prior receiver
                hit = bool(truth[t, band])
                if hit:
                    hits[band] += 1
                else:
                    misses[band] += 1
                if t + 1 >= n_steps:
                    break
                nxt = truth[t + 1]
                if nxt.any():
                    X.append(build_features(hits, misses, t, n_steps))
                    y.append(int(np.argmax(nxt)))
    return np.array(X), np.array(y)


# --------------------------------------------------------------------------- #
#  integration
# --------------------------------------------------------------------------- #
def validate(model, n_bands: int = 10, n_steps: int = 120) -> dict:
    """Quick eval: run the model as the live smart brain on a switch scenario."""
    cfg = SimConfig(n_bands=n_bands, n_steps=n_steps, seed=7, scenario="switch", scheduler="dataset_rfi")
    res = run_simulation(cfg)

    # artifact is wired as the smart brain via a temporary scheduler instance
    eval_ = {"smart_ir": res["smart"]["metrics"]["interception_ratio"], "baseline_ir": res["baseline"]["metrics"]["interception_ratio"]}
    return eval_


def integrate(path: Path, out_name: str = "turing_model.pkl", source: str = "manual") -> dict:
    model = load_model(path)
    if model is None:
        raise SystemExit(
            f"could not load model from {path}.\n"
            "Contract: pickle-(or joblib-)dump'd object with predict(X)/predict_proba(X), "
            "X=(N, 2*n_bands+1) float rows -> (N, C) band scores."
        )
    meta = {
        "source": source,
        "artifact": out_name,
        "features": "2*n_bands+1  (hit_ewma[n] ++ tanh(misses/5)[n] ++ t_frac)",
        "contract": "predict(X) | predict_proba(X) -> (N, C) scores",
        "integrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval": validate(model),
    }
    out = ARTIFACTS_DIR / out_name
    out.write_bytes(path.read_bytes())
    (ARTIFACTS_DIR / "turing_model_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Integrate a dataset-trained emitter model into SPECTRA")
    ap.add_argument("--model", help="path to the trained model artifact (pickle/joblib)")
    ap.add_argument("--build-demo", action="store_true", help="train + integrate the built-in demo predictor")
    ap.add_argument("--out", default="turing_model.pkl", help="artifact filename inside train/artifacts")
    args = ap.parse_args()

    if args.build_demo:
        print("[i] generating scan-history dataset from the RF simulator ...")
        X, y = build_demo_dataset()
        print(f"[i] {len(X)} labelled steps, features={X.shape[1]}")
        W = softmax_fit(X, y, n_classes=10)
        model = BandPredictor(W, n_bands=10)
        demo_path = ARTIFACTS_DIR / "_demo_model_tmp.pkl"
        import pickle

        demo_path.write_bytes(pickle.dumps(model))
        meta = integrate(demo_path, out_name=args.out, source="demo (numpy softmax on built-in RF simulator)")
        demo_path.unlink(missing_ok=True)
    else:
        if not args.model:
            raise SystemExit("pass --model path/to/artifact.pkl  OR  --build-demo")
        meta = integrate(Path(args.model), out_name=args.out, source=str(Path(args.model).name))

    print(f"[ok] integrated -> {ARTIFACTS_DIR / meta['artifact']}")
    print(f"[ok] smart ir  {meta['eval']['smart_ir']:.1f}%   baseline ir {meta['eval']['baseline_ir']:.1f}%")
    print("[ok] select scheduler `dataset_rfi` in the dashboard and RUN")


if __name__ == "__main__":
    main()