"""Choose live model analysis or a lightweight saved sample before importing models."""
import argparse
import errno
from pathlib import Path
import socket
import sys

from nisar.app.paths import FRONTEND_ROOT, SAMPLE_ROOT
from nisar.features.analysis.artifacts import NAMES
from nisar.features.samples.store import SampleSession, load_sample


def choose_mode(input_fn=input):
    while True:
        answer = input_fn("Can this machine run the SAR detector? For <16 GB RAM and no GPU, choose no. [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            return "model"
        if answer in {"", "n", "no"}:
            return "sample"
        print("Enter yes or no.")


def check_listen_address(host, port):
    """Reject an occupied port before spending time loading model weights."""
    try:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.create_server((host, port), family=family):
            pass
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise ValueError(
                f"Port {port} is already in use at {host}. Your browser may be reaching an older "
                "saved-sample or live server. Stop that server, or choose a different port "
                f"with --port {port + 1 if port < 65535 else 8000}.") from exc
        raise ValueError(f"Cannot listen at {host}:{port}: {exc}") from exc


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
        try:
            check_listen_address(args.host, args.port)
        except ValueError as exc:
            parser.exit(2, f"Cannot start live model mode: {exc}\n")
        print(f"Starting live model mode at http://{args.host}:{args.port}; GCOV uploads will be enabled after the detector loads.", flush=True)
        from nisar.app.cli import main as run_models
        sys.argv = ["inference.py", "--host", args.host, "--port", str(args.port),
                    "--sample-dir", str(args.cache_dir), *extra]
        return run_models()
    from nisar.features.chat.qwen import add_chat_options, configure_chat
    chat_parser = argparse.ArgumentParser(prog=parser.prog, add_help=False)
    add_chat_options(chat_parser)
    chat_args, remaining = chat_parser.parse_known_args(extra)
    if remaining:
        parser.error("Detector options require --mode model.")
    try:
        check_listen_address(args.host, args.port)
    except ValueError as exc:
        parser.exit(2, f"Cannot start saved sample mode: {exc}\n")
    try:
        sample = load_sample(args.cache_dir)
        if sample is None:
            raise ValueError("No saved analysis found. Run python cache_sample.py on a model-capable machine first.")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Cannot open saved sample: {exc}\n")
    import uvicorn
    from nisar.app.server import create_app
    chat, unavailable = configure_chat(chat_args)
    app = create_app(SampleSession(sample), FRONTEND_ROOT, NAMES, sample=sample, model_enabled=False,
                     chat=chat, chat_unavailable=unavailable)
    print(f"Saved sample mode: no models or HDF5 inputs loaded. Open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
