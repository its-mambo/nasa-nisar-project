"""Serve the local NISAR/Qwen change analysis application."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from nisar.app.paths import PROJECT_ROOT, FRONTEND_ROOT, MODEL_ROOT
from nisar.features.analysis.service import Demo, NAMES
from nisar.features.reporting.roi_visual import QWEN_ALIGNMENT
from nisar.infrastructure.models.dual_model import (
    QwenModel, SarModel, attach_adapter, convert_weights, open_weights, pack_single_weights,
)


WEIGHT_COMMANDS = {
    "pack": "Export one safetensors file from a .nisar archive or model directory.",
    "convert": "Convert a legacy .nisar archive to a safetensors model directory.",
    "attach-adapter": "Add a PEFT LoRA to a legacy .nisar archive.",
}


def weights_cli(command, argv):
    p = argparse.ArgumentParser(prog=f"{Path(sys.argv[0]).name} {command}",
                                description=WEIGHT_COMMANDS[command])
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    if command == "pack":
        p.add_argument("--work-dir", type=Path,
                       help="Existing directory for temporary conversion files.")
    if command == "attach-adapter":
        p.add_argument("--adapter", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        if command == "pack":
            pack_single_weights(args.weights, args.out, args.work_dir)
        elif command == "convert":
            convert_weights(args.weights, args.out)
        else:
            attach_adapter(args.weights, args.adapter, args.out)
            return
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        p.exit(2, f"{command} failed: {exc}\n")
    print(f"Saved {args.out}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in WEIGHT_COMMANDS:
        return weights_cli(sys.argv[1], sys.argv[2:])
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Weight tools: inference.py pack, convert, or attach-adapter. Add --help for options.")
    p.add_argument("--port",type=int,default=8000)
    p.add_argument("--host",default="127.0.0.1",
                   help="Listen address; use 0.0.0.0 for a trusted LAN, or keep localhost with Tailscale Serve.")
    p.add_argument("--sar-device","--device",default="auto")
    p.add_argument("--qwen-device",default="auto",help="Auto uses a second CUDA GPU when available.")
    p.add_argument("--serial-vision",action="store_true",
                   help="Disable early vision and encode all three views in one pass for comparison.")
    single_default = MODEL_ROOT / "model.safetensors"
    # Support existing hosts that keep their bundle at the old root location.
    if not single_default.is_file() and (PROJECT_ROOT / "model.safetensors").is_file():
        single_default = PROJECT_ROOT / "model.safetensors"
    legacy_default = PROJECT_ROOT / "nisar_seg_model_round2.nisar"
    p.add_argument("--weights", type=Path,
                   default=legacy_default if legacy_default.is_file() and not single_default.is_file() else single_default,
                   help="Model file or directory (defaults to models/model.safetensors).")
    p.add_argument("--roi",nargs=4,type=int,metavar=("X0","Y0","X1","Y1"),
                   help="Run one selection in native T0 pixels and exit; requires --t0, --t1 and --output-dir.")
    p.add_argument("--output-dir",type=Path,help="New directory for the one-shot JSON report and PNGs.")
    p.add_argument("--decoder-overview-size",type=int,metavar="PIXELS",
                   help="Longest Qwen panel edge (default: 512 with LoRA, 768 with base).")
    p.add_argument("--inspect-decoder-tensors",action="store_true",
                   help="Also make the visual token tensors for the latest result available as a NumPy download.")
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
    if args.decoder_overview_size is not None and (
            not 128 <= args.decoder_overview_size <= 1024 or args.decoder_overview_size % QWEN_ALIGNMENT):
        p.error("--decoder-overview-size must be a multiple of 32 from 128 to 1024.")
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
    sar_device=device(args.sar_device)
    if args.qwen_device=="auto" and sar_device.type=="cuda" and torch.cuda.device_count()>1:
        qwen_device=torch.device("cuda",next(i for i in range(torch.cuda.device_count()) if i!=sar_device.index))
    elif args.qwen_device=="auto":
        qwen_device=sar_device
    else:
        qwen_device=device(args.qwen_device)
    with tempfile.TemporaryDirectory(prefix="nisar-dual-",dir=args.work_dir) as folder:
        print(f"Loading persistent models: SAR on {sar_device}, Qwen on {qwen_device}…",flush=True)
        sar_path,qwen_path,adapter=open_weights(args.weights,folder)
        if args.decoder_overview_size is None:
            args.decoder_overview_size=512 if adapter is not None else 768
        sar=SarModel(sar_path,str(sar_device))
        qwen=QwenModel(qwen_path,str(qwen_device),adapter)
        print(f"Qwen decoder: {'bundled LoRA' if qwen.has_adapter else 'base'} "
              f"(panel edge {args.decoder_overview_size})",flush=True)
        demo=Demo(folder,args,sar,qwen)
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
                result.pop("decoder_inputs_url",None)
                if demo.decoder_trace is not None:
                    trace=demo.decoder_trace
                    (args.output_dir/"decoder_trace.json").write_text(
                        json.dumps(dict(attempts=trace.attempts,streams=trace.streams),
                                   ensure_ascii=False,indent=2),encoding="utf-8")
                    if trace.tensor_archive is not None:
                        (args.output_dir/"decoder_tokens.npz").write_bytes(trace.tensor_archive)
                (args.output_dir/"report.json").write_text(
                    json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
            except (OSError,ValueError,KeyError) as e:
                p.exit(2,f"Cannot complete one-shot inference: {e}\n")
            print(f"Saved {args.output_dir/'report.json'}",flush=True)
            if result["interpretation"]["source"]!="visual":
                p.exit(1,"Qwen generation failed; saved the measured report and images. See the log above.\n")
            print(result["interpretation"]["summary"],flush=True)
            return
        import uvicorn
        from nisar.app.server import create_app
        app = create_app(demo, FRONTEND_ROOT, NAMES)
        print(f"Open http://{args.host}:{args.port} · Temporary uploads: {folder}",flush=True)
        uvicorn.run(app,host=args.host,port=args.port,workers=1)


if __name__ == "__main__":
    main()
