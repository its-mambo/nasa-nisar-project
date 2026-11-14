"""Choose live model analysis or a lightweight saved sample before importing models."""
import argparse
from pathlib import Path
import sys

from nisar.app.paths import FRONTEND_ROOT, SAMPLE_ROOT
from nisar.features.analysis.artifacts import NAMES
from nisar.features.samples.store import SampleSession, load_sample


def choose_mode(input_fn=input):
    while True:
        answer = input_fn("Can this machine run the models? For <16 GB RAM and no GPU, choose no. [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            return "model"
        if answer in {"", "n", "no"}:
            return "sample"
        print("Enter yes or no.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ask", "model", "sample"), default="ask")
    parser.add_argument("--cache-dir", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args, extra = parser.parse_known_args()
    mode = args.mode
    if mode == "ask":
        try:
            mode = choose_mode() if sys.stdin.isatty() else "sample"
        except EOFError:
            mode = "sample"
    if mode == "model":
        from nisar.app.cli import main as run_models
        sys.argv = ["inference.py", "--host", args.host, "--port", str(args.port),
                    "--sample-dir", str(args.cache_dir), *extra]
        return run_models()
    if extra:
        parser.error("Model options require --mode model.")
    try:
        sample = load_sample(args.cache_dir)
        if sample is None:
            raise ValueError("No saved analysis found. Run python cache_sample.py on a model-capable machine first.")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Cannot open saved sample: {exc}\n")
    import uvicorn
    from nisar.app.server import create_app
    app = create_app(SampleSession(sample), FRONTEND_ROOT, NAMES, sample=sample, model_enabled=False)
    print(f"Saved sample mode: no models or HDF5 inputs loaded. Open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
