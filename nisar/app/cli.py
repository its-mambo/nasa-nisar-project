"""Serve the local NISAR SAR change analysis application."""
import argparse
import json
import math
from pathlib import Path
import shutil
import tempfile

from nisar.app.paths import FRONTEND_ROOT, MODEL_ROOT, SAMPLE_ROOT
from nisar.features.analysis.service import Demo, NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int,default=8000)
    p.add_argument("--sample-dir", type=Path, default=SAMPLE_ROOT,
                   help="Saved sample offered as optional Step 0 alongside live analysis.")
    p.add_argument("--host",default="127.0.0.1",
                   help="Listen address; use 0.0.0.0 for a trusted LAN, or keep localhost with Tailscale Serve.")
    p.add_argument("--sar-device","--device",default="auto")
    p.add_argument("--weights", type=Path, default=MODEL_ROOT / "detector.safetensors",
                   help="Detector-only checkpoint (defaults to models/detector.safetensors).")
    from nisar.features.chat.qwen import add_chat_options, configure_chat
    add_chat_options(p)
    p.add_argument("--roi",nargs=4,type=int,metavar=("X0","Y0","X1","Y1"),
                   help="Run one selection in native T0 pixels and exit; requires --t0, --t1 and --output-dir.")
    p.add_argument("--output-dir",type=Path,help="New directory for the one-shot JSON report and PNGs.")
    p.add_argument("--threshold",type=float,default=.1)
    p.add_argument("--overlap",type=int,default=32)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--max-delta-db",type=float,default=10)
    p.add_argument("--max-roi-pixels",type=int,help="Optional selection limit in native pixels (default: no cap).")
    p.add_argument("--max-upload-gib",type=int,default=8)
    p.add_argument("--work-dir",type=Path,help="Existing directory on a drive with room for both uploads.")
    p.add_argument("--t0",type=Path,help="Read a local T0 HDF5 directly, without uploading or copying.")
    p.add_argument("--t1",type=Path,help="Read a local T1 HDF5 directly, without uploading or copying.")
    args = p.parse_args()
    if (args.t0 is None) != (args.t1 is None):
        p.error("Provide both --t0 and --t1, or neither.")
    if not (0<args.threshold<1 and 0<=args.overlap<224 and math.isfinite(args.max_delta_db)
            and args.max_delta_db>0 and min(args.batch_size,args.max_upload_gib)>0
            and (args.max_roi_pixels is None or args.max_roi_pixels>0)):
        p.error("Invalid threshold, overlap, batch size, or size/opacity limits.")
    if (args.roi is None) != (args.output_dir is None) or (args.roi is not None and args.t0 is None):
        p.error("--roi requires --output-dir and both local input files.")
    if args.output_dir is not None and (args.output_dir.exists() or args.output_dir.is_symlink()):
        p.error("--output-dir must be a new path.")
    import torch
    def device(name):
        d=torch.device(name if name!="auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        return torch.device("cuda",torch.cuda.current_device()) if d.type=="cuda" and d.index is None else d
    from nisar.infrastructure.models.change_model import SarModel

    sar_device = device(args.sar_device)
    with tempfile.TemporaryDirectory(prefix="nisar-detector-",dir=args.work_dir) as folder:
        print(f"Loading SAR detector on {sar_device}…", flush=True)
        try:
            sar = SarModel(args.weights, str(sar_device))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            p.exit(2, f"Cannot load detector: {exc}\n")
        demo = Demo(folder, args, sar)
        if args.t0 is not None:
            demo.local_paths={"t0":args.t0.resolve(),"t1":args.t1.resolve()}
            try:
                demo.prepare()
            except (OSError,ValueError,KeyError) as e:
                p.exit(2,f"Cannot prepare local pair: {e}\n")
        if args.roi is not None:
            try:
                result=demo.infer(args.roi,demo.info["revision"])
                args.output_dir.mkdir(parents=True,exist_ok=False)
                for name in NAMES:
                    shutil.copyfile(demo.dir/name,args.output_dir/name)
                result["images"]={name:name for name in NAMES}
                result.pop("viewer_url",None)
                (args.output_dir/"report.json").write_text(
                    json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
            except (OSError,ValueError,KeyError) as e:
                p.exit(2,f"Cannot complete one-shot inference: {e}\n")
            print(f"Saved {args.output_dir/'report.json'}",flush=True)
            return
        import uvicorn
        from nisar.app.server import create_app
        from nisar.features.samples.store import load_sample
        sample_unavailable = None
        try:
            sample = load_sample(args.sample_dir)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Saved sample unavailable: {exc}", flush=True)
            sample = None
            sample_unavailable = f"The saved analysis could not be loaded: {exc}"
        chat, unavailable = configure_chat(args, default_device=sar_device)
        app = create_app(demo, FRONTEND_ROOT, NAMES, sample=sample, sample_unavailable=sample_unavailable,
                         chat=chat, chat_unavailable=unavailable)
        print(f"Open http://{args.host}:{args.port} · Temporary uploads: {folder}",flush=True)
        uvicorn.run(app,host=args.host,port=args.port,workers=1)


if __name__ == "__main__":
    main()
