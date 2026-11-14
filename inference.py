#!/usr/bin/env python3
"""Serve the local NISAR/Qwen change analysis application."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from time import perf_counter
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image
from roi_interpretation import interpret, radar_patterns
from roi_report import paired_sector_deltas, report_sections, scene_observations
from roi_visual import DecoderTrace, QWEN_ALIGNMENT, native_views, native_change_map
from roi_interpretation import LIMIT
from dual_model import (QwenModel, SarModel, attach_adapter, convert_weights,
                        open_weights, pack_single_weights)

ROOT = Path(__file__).resolve().parent
NAMES = ("t0.png", "t1.png", "change.png")
PREVIEW_TILE = 512


def tone(z, valid, limits, *, grayscale=False, soft_color=False):
    lo,hi = np.asarray(limits, dtype=np.float32).T[:,:,None,None]
    pixels = np.zeros((*valid.shape,4), np.uint8)
    scaled = np.clip((np.where(valid[None],z,lo)-lo)/(hi-lo),0,1)
    if soft_color:
        # Muted display tones for the preview, avoiding bright yellow highlights.
        scaled *= .75
        scaled += .08
    pixels[:,:,:2] = np.rint(scaled*255).astype(np.uint8).transpose(1,2,0)
    if grayscale:
        pixels[:,:,:3] = pixels[:,:,:1]
    pixels[:,:,3] = valid.astype(np.uint8)*255
    return Image.fromarray(pixels)


def limits_for(z,v,default):
    if not v.any():
        return default
    # Downweight very dark strips that can wash out the rest of the swath.
    limits = np.percentile(z[:,v], [10,98], axis=1).T
    return np.where((limits[:,1]>limits[:,0])[:,None], limits, default)


def medians(z,valid):
    if not valid.any():
        return None
    return [round(float(x),2) for x in np.median(z[:,valid],axis=1)]


def dates_note(m0,m1):
    warnings = [f"Acquisitions differ in {k}; check comparability before interpreting changes."
                for k in ("orbitPassDirection","lookDirection","trackNumber","relativeOrbitNumber")
                if k in m0 and k in m1 and m0[k] != m1[k]]
    if m0.get("zeroDopplerStartTime","") > m1.get("zeroDopplerStartTime","") and m1.get("zeroDopplerStartTime"):
        warnings.append("T0 acquisition starts after T1. Check input order.")
    return warnings


def observe(z0,z1,valid,selected):
    names = ("upper left","upper center","upper right","middle left","center",
             "middle right","lower left","lower center","lower right")
    h,w = valid.shape
    zones = []
    for i,name in enumerate(names):
        r,c = divmod(i,3)
        s = np.s_[r*h//3:(r+1)*h//3,c*w//3:(c+1)*w//3]
        v = valid[s]
        n = int(v.sum())
        mean = float((z0[0][s][v].mean(dtype=np.float64)+z1[0][s][v].mean(dtype=np.float64))/2) if n else None
        zones.append(dict(zone=name,valid=n,flagged=int((selected[s]&v).sum()),mean_hh_db=mean))
    return scene_observations(zones)


class Demo:
    def __init__(self, directory, options, sar, qwen):
        self.dir, self.options, self.spec = Path(directory), options, sar.spec
        self.sar, self.qwen = sar,qwen
        spec=sar.spec
        self.frequency = spec["frequency"]
        self.spacing = float(spec["spacing_m"])
        self.clip = np.asarray([spec["db_clip"]]*2)
        self.info = None
        self.lock = threading.Lock()
        self.counter = 0
        self.result_revision = None
        self.decoder_trace = None
        self.local_paths = {}
        self.pending_uploads = {}

    def path(self,date):
        return self.local_paths.get(date,self.dir/f"{date}.h5")

    def upload(self,date,offset,total,data):
        import h5py
        from gcov import Grid

        if not 0 < total <= self.options.max_upload_gib * 1024**3:
            raise ValueError(f"{date.upper()} exceeds the per-file limit (--max-upload-gib).")
        if not 0 <= offset < offset+len(data) <= total:
            raise ValueError("Invalid upload byte range.")
        tmp = self.dir/f"{date}.partial"
        if offset == 0:
            self.pending_uploads.pop(date,None)
            tmp.unlink(missing_ok=True)
            if shutil.disk_usage(self.dir).free < total + 100*1024**2:
                raise ValueError(f"Insufficient temporary disk space for {date.upper()}; use --work-dir on a larger drive.")
            self.pending_uploads[date] = (total,0)
        if self.pending_uploads.get(date) != (total,offset):
            raise ValueError(f"{date.upper()} upload offset mismatch. Restart the upload.")
        end = offset+len(data)
        try:
            with tmp.open("ab") as dst:
                dst.write(data)
            if end == total:
                print(f"Validating {date.upper()} HDF5 ({total:,} bytes)…",flush=True)
                with h5py.File(tmp,"r") as product:
                    Grid.read(product,self.frequency)
                tmp.replace(self.dir/f"{date}.h5")
                self.local_paths.pop(date,None)
                self.pending_uploads.pop(date,None)
                self.info = None
            else:
                self.pending_uploads[date] = (total,end)
        except Exception:
            self.pending_uploads.pop(date,None)
            tmp.unlink(missing_ok=True)
            raise
        return {"received":end,"complete":end==total}

    def prepare(self):
        import h5py
        from gcov import Grid, db, factor, match, metadata, move

        self.info = None
        if self.pending_uploads:
            raise ValueError("An upload is incomplete. Upload both files before preparing.")
        if not all(self.path(date).is_file() for date in ("t0","t1")):
            raise ValueError("Choose both T0 and T1 HDF5 products.")
        print("Preparing GCOV metadata and preview…",flush=True)
        with h5py.File(self.path("t0"),"r") as h0,h5py.File(self.path("t1"),"r") as h1:
            a,b = (Grid.read(h,self.frequency) for h in (h0,h1))
            shift,common = match(a,b)
            factor(a,self.spacing)
            stride = max(1,math.ceil(max(a.width,a.height)/1400))
            power,valid = a.power(0,0,a.width,a.height,stride)
            overview = db(power)
            limits = limits_for(overview,valid,self.clip)
            preview = tone(overview,valid,limits,soft_color=True)
            preview.save(self.dir/"preview.png")
            tone(overview,valid,limits,grayscale=True).save(self.dir/"preview_hh.png")
            step = max(1,math.ceil(max(common[2]-common[0],common[3]-common[1])/500))
            p0,v0 = a.power(*common,step=step)
            p1,v1 = b.power(*move(common,shift),step=step)
            valid = v0&v1
            m0,m1 = metadata(h0),metadata(h1)
            self.counter += 1
            self.info = dict(revision=self.counter, width=a.width,height=a.height,stride=stride,
                             preview_tile_size=PREVIEW_TILE,
                             preview_width=preview.width,preview_height=preview.height,
                             preview_url=f"/preview.png?v={self.counter}",
                             preview_hh_url=f"/preview_hh.png?v={self.counter}",
                             overlap=common, epsg=a.epsg,
                             native_spacing_m=[a.dx,a.dy],model_spacing_m=self.spacing,frequency=self.frequency,
                             acquisitions={"t0":m0,"t1":m1}, warnings=dates_note(m0,m1),
                             display_ranges_db=limits.tolist(),
                             full_scene_context=dict(scope="Systematic sample of common footprint; no full-scene inference",
                                  sample_stride=step, sample_pairs=int(valid.sum()),
                                  median_t0_hh_hv_db=medians(db(p0),valid),
                                  median_t1_hh_hv_db=medians(db(p1),valid)))
        return self.info

    def preview_tile(self, mode, step, col, row):
        import h5py
        from gcov import Grid, db

        info = self.info
        if (mode not in ("rg", "hh") or step < 1 or step > info["stride"]
                or step & (step - 1) or col < 0 or row < 0):
            raise ValueError("Invalid preview tile.")
        span = PREVIEW_TILE * step
        x, y = col * span, row * span
        width, height = info["width"], info["height"]
        if x >= width or y >= height:
            raise ValueError("Preview tile is outside the grid.")
        with h5py.File(self.path("t0"), "r") as product:
            grid = Grid.read(product, self.frequency)
            power, valid = grid.power(x, y, min(x + span, width), min(y + span, height), step)
        return tone(db(power), valid, info["display_ranges_db"], grayscale=mode == "hh", soft_color=mode == "rg")

    def infer(self, box, revision):
        import h5py
        from gcov import Grid, db, factor, match, move, roi_box, to_model

        started=perf_counter()
        if self.info is None or revision != self.info["revision"]:
            raise ValueError("Prepare the current pair before selecting an area.")
        opts = self.options
        with h5py.File(self.path("t0"),"r") as h0,h5py.File(self.path("t1"),"r") as h1:
            a,b = (Grid.read(h,self.frequency) for h in (h0,h1))
            shift,common = match(a,b)
            box = roi_box(box,a,common,opts.max_roi_pixels)
            fy,fx = factor(a,self.spacing)
            p0,v0 = a.power(*box)
            p1,v1 = b.power(*move(box,shift))
            bounds = a.bounds(box)
            t1_box = move(box,shift)
            rtc = [a.rtc_median(box,v0&v1),b.rtc_median(t1_box,v0&v1)]
        h,w = v0.shape
        native_valid = v0 & v1
        if not native_valid.any():
            raise ValueError("No jointly valid HH/HV data in this selection.")
        # The inference path uses float power averaged at training spacing, then dB.
        z0,m0 = to_model(p0,v0,fy,fx)
        z1,m1 = to_model(p1,v1,fy,fx)
        model_valid = m0&m1
        if not model_valid.any():
            raise ValueError("No complete, jointly valid model cells. Choose a larger selection.")
        mh,mw = model_valid.shape

        def pair(window):
            x0,y0,x1,y1 = window
            s = np.s_[y0:y1,x0:x1]
            return (z0[:,s[0],s[1]],m0[s]),(z1[:,s[0],s[1]],m1[s])

        pictures,edges=native_views(z0,z1,m0,m1,self.clip,opts.decoder_overview_size)
        ready=perf_counter()
        parallel=(not opts.serial_vision and self.sar.device.type==self.qwen.device.type=="cuda"
                  and self.sar.device!=self.qwen.device)
        cached_pair=None
        with ThreadPoolExecutor(max_workers=1) as workers:
            pending=workers.submit(self.qwen.encode,pictures) if parallel else None
            mask,_,valid_count,changed=self.sar.detect((mw,mh),pair,batch_size=opts.batch_size,
                                                        overlap=opts.overlap,threshold=opts.threshold)
            result=self.sar.classify((mw,mh),pair,mask)
            spatial=observe(z0,z1,model_valid,mask)
            paired=paired_sector_deltas(z0,z1,model_valid,mask,spatial["zones"])
            pattern_evidence=radar_patterns(z0,z1,model_valid,mask)
            if pending is not None:
                try:
                    cached_pair=pending.result()
                except (RuntimeError,ValueError,TypeError,AttributeError):
                    logging.exception("Early Qwen vision failed; using the native three-panel path.")
        pictures.append(native_change_map(pictures[0].size,edges,mask,model_valid))
        detected=perf_counter()
        del pair,z0,z1,m0,m1,model_valid
        a0,a1 = db(p0),db(p1)
        del p0,p1
        roi_medians = [medians(a0,native_valid),medians(a1,native_valid)]
        full = self.info["full_scene_context"]
        result.update(threshold=opts.threshold,valid_pixels=valid_count,changed_pixels=changed,
                      spatial_context=spatial, full_scene_context=full,paired_sector_deltas=paired,
                      roi=dict(t0_pixel_box=box,t1_pixel_box=t1_box,width_px=w,height_px=h,
                               projected_bounds_wsen=bounds,epsg=self.info["epsg"],
                               projected_rectangle_km2=round(w*h*math.prod(self.info["native_spacing_m"])/1e6,6),
                               native_joint_valid_pixels=int(native_valid.sum()),
                               native_spacing_m=self.info["native_spacing_m"],model_spacing_m=self.spacing,
                               model_grid_width_px=mw,model_grid_height_px=mh),
                      roi_measurements=dict(median_hh_hv_db_t0_t1=roi_medians,
                                            median_rtc_gamma_to_sigma_factor_t0_t1=rtc),
                      acquisitions=self.info["acquisitions"],warnings=self.info["warnings"])
        result["report_sections"] = report_sections(result)
        result["pattern_evidence"] = pattern_evidence
        trace=DecoderTrace(export_tensors=opts.inspect_decoder_tensors)
        report_started=perf_counter()
        try:
            raw=self.qwen.report(pictures,result,trace,cached_pair)
            if not raw.strip():
                raise ValueError("Qwen returned an empty caption.")
            result["interpretation"]=dict(source="visual",summary=raw,limitation=LIMIT,
                                          decoder="qwen_lora" if self.qwen.has_adapter else "qwen_base")
        except (RuntimeError,ValueError,TypeError,AttributeError):
            logging.exception("Qwen interpretation unavailable")
            result["interpretation"]=interpret(pattern_evidence,result,lambda _: [])
            result["interpretation"]["visual_note"]="Qwen interpretation unavailable; showing measured evidence."
        result["timing_s"]={"input_and_render":round(ready-started,3),
                            "detection_classification_and_early_vision":round(detected-ready,3),
                            "qwen_report":round(perf_counter()-report_started,3)}
        # Back-project the coarse binary mask by integer pixel replication, then
        # render native HH dB differences only in flagged, jointly valid pixels.
        selected = np.repeat(np.repeat(mask,fy,axis=0),fx,axis=1)[:h,:w] & native_valid
        blue = np.zeros((h,w,4),np.uint8)
        blue[:,:,:3] = (0,120,255)
        delta = np.abs(a1[0]-a0[0])
        blue[:,:,3] = np.rint(np.clip(np.where(selected,delta,0)/opts.max_delta_db,0,1)*255).astype(np.uint8)
        display = self.info["display_ranges_db"]
        t0 = tone(a0,v0,display)
        t1 = tone(a1,v1,display)
        self.result_revision = None
        for name,img in zip(NAMES,(t0,t1,Image.alpha_composite(t0,Image.fromarray(blue))),strict=True):
            tmp = self.dir/(name+".tmp")
            img.save(tmp,format="PNG")
            tmp.replace(self.dir/name)
        self.counter += 1
        self.result_revision = self.counter
        self.decoder_trace = trace if trace is not None and trace.attempts else None
        if self.decoder_trace is not None:
            result["decoder_inputs_url"] = f"/decoder-inputs?v={self.counter}"
        result["images"] = {n: f"/{n}?v={self.counter}" for n in NAMES}
        result["viewer_url"] = f"/viewer?v={self.counter}"
        result["timing_s"]["total"]=round(perf_counter()-started,3)
        return result


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, data, content_type):
        self.send_response(code)
        self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(data)))
        self.send_header("Cache-Control","no-store")
        self.end_headers()
        self.wfile.write(data)

    def json(self,code,payload):
        self.respond(code,json.dumps(payload,allow_nan=False).encode(),"application/json; charset=utf-8")

    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        if path in ("/","/viewer"):
            name = "viewer.html" if path == "/viewer" else "index.html"
            self.respond(200,(ROOT/name).read_bytes(),"text/html; charset=utf-8")
        elif path == "/zoom.js":
            self.respond(200,(ROOT/"zoom.js").read_bytes(),"text/javascript; charset=utf-8")
        elif path == "/pair":
            self.json(200,{"pair":self.server.demo.info})
        elif path in ("/decoder-inputs","/decoder-panel.png","/decoder-tokens.npz"):
            demo = self.server.demo
            query = parse_qs(url.query)
            with demo.lock:
                revision,trace = demo.result_revision,demo.decoder_trace
            if revision is None or query.get("v") != [str(revision)]:
                self.send_error(410,"This result is no longer available")
            elif trace is None:
                self.send_error(404,"No visual decoder inputs for this result")
            elif path == "/decoder-inputs":
                self.json(200,trace.manifest(revision))
            elif path == "/decoder-tokens.npz":
                if trace.tensor_archive is None:
                    self.send_error(404,"Tensor capture is disabled")
                else:
                    self.respond(200,trace.tensor_archive,"application/octet-stream")
            else:
                try:
                    i = int(query["i"][0])
                    if not 0 <= i < len(trace.pngs):
                        raise ValueError
                except (KeyError,ValueError):
                    self.send_error(400,"Invalid decoder panel")
                    return
                self.respond(200,trace.pngs[i],"image/png")
        elif path == "/preview-tile.png":
            q = parse_qs(url.query)
            demo = self.server.demo
            try:
                with demo.lock:
                    if demo.info is None or q.get("v") != [str(demo.info["revision"])]:
                        self.send_error(410,"This preview is no longer available")
                        return
                    mode, step, col, row = (q[key][0] for key in ("mode", "step", "col", "row"))
                    tile = demo.preview_tile(mode,int(step),int(col),int(row))
                with BytesIO() as out:
                    tile.save(out,format="PNG",compress_level=1)
                    self.respond(200,out.getvalue(),"image/png")
            except (KeyError, ValueError):
                self.send_error(400,"Invalid preview tile")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError:
                self.send_error(500,"Could not read the HDF5 preview tile")
        elif path in ("/preview.png","/preview_hh.png",*("/"+n for n in NAMES)):
            try:
                demo = self.server.demo
                if path[1:] in NAMES:
                    with demo.lock:
                        revision = parse_qs(url.query).get("v",[None])[0]
                        if demo.result_revision is None or revision != str(demo.result_revision):
                            self.send_error(410,"This result is no longer available")
                            return
                        image = (demo.dir/path[1:]).open("rb")
                else:
                    image = (demo.dir/path[1:]).open("rb")
                with image as f:
                    self.send_response(200)
                    self.send_header("Content-Type","image/png")
                    self.send_header("Cache-Control","no-store")
                    self.end_headers()
                    shutil.copyfileobj(f,self.wfile)
            except FileNotFoundError:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.headers.get("X-Nisar-Demo") != "1":
            self.json(403,{"error":"Open the local demo page to submit requests."})
            return
        demo = self.server.demo
        try:
            with demo.lock:
                if self.path in ("/upload/t0","/upload/t1"):
                    date = self.path[-2:]
                    size = int(self.headers.get("Content-Length","0"))
                    if not 0 < size <= 8*1024**2:
                        raise ValueError("Expected an upload chunk of at most 8 MiB. Refresh the demo page.")
                    offset = int(self.headers.get("X-Upload-Offset","-1"))
                    total = int(self.headers.get("X-Upload-Size","0"))
                    data = self.rfile.read(size)
                    if len(data) != size:
                        raise ValueError(f"{date.upper()} upload disconnected at byte {offset+len(data):,}.")
                    response = demo.upload(date,offset,total,data)
                elif self.path == "/prepare":
                    response = demo.prepare()
                elif self.path == "/infer":
                    n = int(self.headers.get("Content-Length","0"))
                    if not 0<n<=2048:
                        raise ValueError("Expected a small selection JSON body.")
                    body = json.loads(self.rfile.read(n))
                    if not isinstance(body,dict):
                        raise ValueError("Expected a selection object.")
                    response = demo.infer(body.get("box"),body.get("revision"))
                else:
                    self.json(404,{"error":"Unknown endpoint."})
                    return
            self.json(200,response)
        except (BrokenPipeError,ConnectionResetError):
            self.log_error("Client disconnected during %s",self.path)
        except KeyError as e:
            self.json(400,{"error":f"Missing HDF5 dataset or checkpoint field: {e}"})
        except (ValueError,OSError) as e:
            self.json(400,{"error":str(e)})
        except Exception as e:
            traceback.print_exc()
            self.json(500,{"error":f"{type(e).__name__}: {e}"})


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
    p.add_argument("--sar-device","--device",default="auto")
    p.add_argument("--qwen-device",default="auto",help="Auto uses a second CUDA GPU when available.")
    p.add_argument("--serial-vision",action="store_true",
                   help="Disable early vision and encode all three views in one pass for comparison.")
    single_default = ROOT / "model.safetensors"
    legacy_default = ROOT / "nisar_seg_model_round2.nisar"
    p.add_argument("--weights", type=Path,
                   default=legacy_default if legacy_default.is_file() and not single_default.is_file() else single_default,
                   help="Model file or directory (defaults to Round 2 weights beside this script).")
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
        with ThreadingHTTPServer(("127.0.0.1",args.port),Handler) as server:
            server.demo=demo
            print(f"Open http://127.0.0.1:{server.server_port} · Temporary uploads: {folder}",flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
