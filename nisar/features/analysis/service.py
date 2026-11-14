"""Coordinate GCOV pair preparation and selection analysis with loaded models."""
from concurrent.futures import ThreadPoolExecutor
import logging
import math
from pathlib import Path
import shutil
import threading
from time import perf_counter

import numpy as np
from PIL import Image

from nisar.features.analysis.artifacts import NAMES, PREVIEW_TILE
from nisar.features.imagery.rendering import aligned_power, limits_for, tone
from nisar.features.reporting.roi_interpretation import LIMIT, interpret, radar_patterns
from nisar.features.reporting.roi_report import paired_sector_deltas, report_sections, scene_observations
from nisar.features.reporting.evidence_report import build_report
from nisar.features.reporting.roi_visual import DecoderTrace, native_views, native_change_map


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
        self.last_result = None
        self.decoder_trace = None
        self.local_paths = {}
        self.pending_uploads = {}

    def path(self,date):
        return self.local_paths.get(date,self.dir/f"{date}.h5")

    def upload(self,date,offset,total,data):
        import h5py
        from nisar.features.imagery.gcov import Grid

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
        from nisar.features.imagery.gcov import Grid, db, factor, match, metadata, move

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
            later, later_valid = aligned_power(b, (0,0,a.width,a.height), shift, stride)
            later = db(later)
            tone(later,later_valid,limits,soft_color=True).save(self.dir/"preview_t1.png")
            tone(later,later_valid,limits,grayscale=True).save(self.dir/"preview_t1_hh.png")
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
                             preview_t1_url=f"/preview_t1.png?v={self.counter}",
                             preview_t1_hh_url=f"/preview_t1_hh.png?v={self.counter}",
                             t1_shift=shift,
                             overlap=common, epsg=a.epsg,
                             native_spacing_m=[a.dx,a.dy],model_spacing_m=self.spacing,frequency=self.frequency,
                             acquisitions={"t0":m0,"t1":m1}, warnings=dates_note(m0,m1),
                             threshold=self.options.threshold,
                             display_ranges_db=limits.tolist(),
                             full_scene_context=dict(scope="Systematic sample of common footprint; no full-scene inference",
                                  sample_stride=step, sample_pairs=int(valid.sum()),
                                  median_t0_hh_hv_db=medians(db(p0),valid),
                                  median_t1_hh_hv_db=medians(db(p1),valid)))
        return self.info

    def preview_tile(self, mode, step, col, row, date="t0"):
        import h5py
        from nisar.features.imagery.gcov import Grid, db

        info = self.info
        if (date not in ("t0", "t1") or mode not in ("rg", "hh") or step < 1 or step > info["stride"]
                or step & (step - 1) or col < 0 or row < 0):
            raise ValueError("Invalid preview tile.")
        span = PREVIEW_TILE * step
        x, y = col * span, row * span
        width, height = info["width"], info["height"]
        if x >= width or y >= height:
            raise ValueError("Preview tile is outside the grid.")
        with h5py.File(self.path(date), "r") as product:
            grid = Grid.read(product, self.frequency)
            box = (x, y, min(x + span, width), min(y + span, height))
            power, valid = aligned_power(grid, box, info["t1_shift"], step) if date == "t1" else grid.power(*box, step)
        return tone(db(power), valid, info["display_ranges_db"], grayscale=mode == "hh", soft_color=mode == "rg")

    def infer(self, box, revision, threshold=None, progress=None):
        import h5py
        from nisar.features.imagery.gcov import Grid, db, factor, match, move, roi_box, to_model

        started=perf_counter()
        if self.info is None or revision != self.info["revision"]:
            raise ValueError("Prepare the current pair before selecting an area.")
        opts = self.options
        threshold = opts.threshold if threshold is None else threshold
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 < threshold < 1):
            raise ValueError("Change detection threshold must be a number between 0 and 1 (exclusive).")
        def update(stage, message, completed=None, total=None):
            if progress is not None:
                progress(stage, message, completed, total)
        update("read", "Reading native pixels and aligning the two dates…")
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
            update("detect", "Starting the change detector…")
            mask,_,valid_count,changed=self.sar.detect((mw,mh),pair,batch_size=opts.batch_size,
                overlap=opts.overlap,threshold=threshold,
                progress=lambda done,total: update("detect", f"Processed {done:,} of {total:,} image tiles", done, total))
            update("classify", "Measuring SAR patterns and classifying flagged regions…")
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
        result.update(threshold=threshold,valid_pixels=valid_count,changed_pixels=changed,
                      analysis_settings=dict(frequency=self.frequency,channels=["HH","HV"],
                          detector_sha256=getattr(self.sar,"detector_sha256",None),
                          detector_tile_size_px=224,detector_overlap_px=opts.overlap,
                          batch_size=opts.batch_size,db_clip=np.asarray(self.spec["db_clip"],dtype=float).tolist(),
                          overlay_max_delta_db=opts.max_delta_db,
                          preprocessing="Mean native power at model spacing, then 10 log10 to dB"),
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
        update("report", "Generating the explanation from both dates and the detector map…")
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
        result["evidence_report"] = build_report(result)
        result["timing_s"]={"input_and_render":round(ready-started,3),
                            "detection_classification_and_early_vision":round(detected-ready,3),
                            "qwen_report":round(perf_counter()-report_started,3)}
        # Back-project the coarse binary mask by integer pixel replication, then
        # render native HH dB differences only in flagged, jointly valid pixels.
        update("render", "Rendering the change overlay and preparing your results…", 0, len(NAMES))
        selected = np.repeat(np.repeat(mask,fy,axis=0),fx,axis=1)[:h,:w] & native_valid
        blue = np.zeros((h,w,4),np.uint8)
        blue[:,:,:3] = (0,120,255)
        delta = np.abs(a1[0]-a0[0])
        blue[:,:,3] = np.rint(np.clip(np.where(selected,delta,0)/opts.max_delta_db,0,1)*255).astype(np.uint8)
        display = self.info["display_ranges_db"]
        overlay = Image.fromarray(blue)
        del blue
        self.result_revision = None
        images_written = 0
        def save_image(name,img):
            nonlocal images_written
            tmp = self.dir/(name+".tmp")
            img.save(tmp,format="PNG")
            tmp.replace(self.dir/name)
            images_written += 1
            update("render", f"Prepared {images_written} of {len(NAMES)} result images", images_written, len(NAMES))
        save_image("overlay.png",overlay)
        t0 = tone(a0,v0,display)
        save_image("t0.png",t0)
        save_image("change.png",Image.alpha_composite(t0,overlay))
        del t0
        t1 = tone(a1,v1,display)
        save_image("t1.png",t1)
        save_image("change_t1.png",Image.alpha_composite(t1,overlay))
        del t1
        t0_hh = tone(a0,v0,display,grayscale=True)
        save_image("t0_hh.png",t0_hh)
        save_image("change_hh.png",Image.alpha_composite(t0_hh,overlay))
        del t0_hh
        t1_hh = tone(a1,v1,display,grayscale=True)
        save_image("t1_hh.png",t1_hh)
        save_image("change_t1_hh.png",Image.alpha_composite(t1_hh,overlay))
        del t1_hh, overlay
        self.counter += 1
        self.result_revision = self.counter
        self.decoder_trace = trace if trace is not None and trace.attempts else None
        if self.decoder_trace is not None:
            result["decoder_inputs_url"] = f"/decoder-inputs?v={self.counter}"
        result["images"] = {n: f"/{n}?v={self.counter}" for n in NAMES}
        result["viewer_url"] = f"/change?v={self.counter}"
        result["timing_s"]["total"]=round(perf_counter()-started,3)
        # Kept so /result can reopen the Results tab (e.g. from the Change page).
        self.last_result = result
        return result
