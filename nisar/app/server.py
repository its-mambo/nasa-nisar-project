"""FastAPI routes for the NISAR analysis demo."""

from io import BytesIO
from collections import deque
import asyncio
import json
from pathlib import Path
import threading
from time import monotonic
import traceback
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from nisar.app.web import register_web_routes
from nisar.features.analysis.progress_timing import RunTiming
from nisar.features.reporting.evidence_report import build_report, render_export
from nisar.features.samples.store import IMAGE_FILES


MAX_CHUNK = 8 * 1024 * 1024


def error(code, message):
    return JSONResponse({"error": message}, status_code=code)


def call_locked(demo, func):
    try:
        with demo.lock:
            return func()
    except KeyError as exc:
        return error(400, f"Missing HDF5 dataset or checkpoint field: {exc}")
    except (ValueError, OSError) as exc:
        return error(400, str(exc))
    except Exception as exc:
        traceback.print_exc()
        return error(500, f"{type(exc).__name__}: {exc}")


def create_app(demo, root: Path, names, *, sample=None, model_enabled=True):
    app = FastAPI(title="NISAR SAR Change Explorer")
    image_names = set(names)
    # Progress has its own lock: inference holds demo.lock while using the models.
    # Status requests must remain responsive throughout that work.
    jobs, job_lock, tasks = {}, threading.Lock(), set()
    timing_history = deque(maxlen=8)

    def job_snapshot(job):
        return {key: value for key, value in job.items() if key not in {"started", "finished", "timing"}} | {
            "elapsed_s": round(job.get("finished", monotonic()) - job["started"], 1),
            "estimate": job["timing"].estimate() if job["status"] == "running" else None}

    async def run_job(run_id, body):
        def update(stage, message, completed=None, total=None):
            with job_lock:
                jobs[run_id]["timing"].update(stage, completed, total)
                jobs[run_id].update(stage=stage, message=message, completed=completed, total=total)

        def work():
            return call_locked(demo, lambda: demo.infer(body.get("box"), body.get("revision"),
                                                       body.get("threshold"), progress=update))

        try:
            result = await run_in_threadpool(work)
            with job_lock:
                job = jobs[run_id]
                if isinstance(result, JSONResponse):
                    job.update(status="failed", error=json.loads(result.body)["error"])
                else:
                    timing_history.append(job["timing"].profile())
                    job.update(status="complete", stage="complete", message="Analysis complete", result=result)
                job["finished"] = monotonic()
        except Exception as exc:
            traceback.print_exc()
            with job_lock:
                jobs[run_id].update(status="failed", error=str(exc), finished=monotonic())

    @app.middleware("http")
    async def no_cache(request, next_call):
        response = await next_call(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    register_web_routes(app, root)

    @app.get("/sample")
    def sample_info():
        return {"model_enabled": model_enabled,
                "sample": sample.describe() if sample else {"available": False}}

    def saved_revision(v):
        return sample is not None and v == sample.revision

    def read_only():
        return error(409, "Saved sample mode is read-only. Start launch.py --mode model to analyze another area.")

    @app.get("/pair")
    def pair():
        with demo.lock:
            # result_revision: latest analysis result (None until a model run finishes).
            return {"pair": demo.info, "result_revision": demo.result_revision}

    @app.get("/result")
    def result(v: str = ""):
        if saved_revision(v) or (not model_enabled and not v and sample is not None):
            return {"result": sample.result(), "result_revision": sample.revision}
        with demo.lock:
            revision = demo.result_revision
            if v and v != str(revision):
                return error(410, "This result is no longer available")
            data = getattr(demo, "last_result", None) if revision is not None else None
        # The UI can adopt an already completed result from before the structured report upgrade.
        if data is not None and "evidence_report" not in data:
            data = {**data,"evidence_report":build_report(data)}
        return {"result": data, "result_revision": revision}

    @app.get("/report.{format}")
    def report_export(format: str, v: str = ""):
        if format not in {"md","txt"}:
            return error(404,"Choose a Markdown or plain-text report.")
        if saved_revision(v):
            data = sample.result()
        else:
            with demo.lock:
                if demo.result_revision is None or v != str(demo.result_revision) or demo.last_result is None:
                    return error(410,"This result is no longer available")
                data = demo.last_result
        contents = render_export(data,format)
        return Response(contents,media_type="text/markdown" if format=="md" else "text/plain",
                        headers={"Content-Disposition":f'attachment; filename="nisar-evidence-report-{v}.{format}"'})

    @app.get("/decoder-inputs")
    def decoder_inputs(v: str = ""):
        if saved_revision(v):
            return error(404, "Live decoder inputs are not included in the portable sample.")
        with demo.lock:
            revision, trace = demo.result_revision, demo.decoder_trace
        if revision is None or v != str(revision):
            return error(410, "This result is no longer available")
        if trace is None:
            return error(404, "No visual decoder inputs for this result")
        return trace.manifest(revision)

    @app.get("/decoder-tokens.npz")
    def decoder_tokens(v: str = ""):
        if saved_revision(v):
            return error(404, "Decoder tensors are not included in the portable sample.")
        with demo.lock:
            revision, trace = demo.result_revision, demo.decoder_trace
        if revision is None or v != str(revision):
            return error(410, "This result is no longer available")
        if trace is None:
            return error(404, "No visual decoder inputs for this result")
        if trace.tensor_archive is None:
            return error(404, "Tensor capture is disabled")
        return Response(trace.tensor_archive, media_type="application/octet-stream")

    @app.get("/decoder-panel.png")
    def decoder_panel(v: str = "", i: str = ""):
        if saved_revision(v):
            return error(404, "Live decoder inputs are not included in the portable sample.")
        with demo.lock:
            revision, trace = demo.result_revision, demo.decoder_trace
        if revision is None or v != str(revision):
            return error(410, "This result is no longer available")
        if trace is None:
            return error(404, "No visual decoder inputs for this result")
        try:
            index = int(i)
            if not 0 <= index < len(trace.pngs):
                raise ValueError
        except ValueError:
            return error(400, "Invalid decoder panel")
        return Response(trace.pngs[index], media_type="image/png")

    @app.get("/preview-tile.png")
    def preview_tile(v: str = "", mode: str = "", step: str = "", col: str = "", row: str = "", date: str = "t0"):
        if not model_enabled:
            return error(404, "Sample mode uses saved images rather than live preview tiles.")
        def render():
            if demo.info is None or v != str(demo.info["revision"]):
                return error(410, "This preview is no longer available")
            tile = demo.preview_tile(mode, int(step), int(col), int(row), date=date)
            with BytesIO() as output:
                tile.save(output, format="PNG", compress_level=1)
                return Response(output.getvalue(), media_type="image/png")
        return call_locked(demo, render)

    @app.get("/{filename}.png")
    def png(filename: str, v: str = ""):
        name = filename + ".png"
        if saved_revision(v):
            if name not in IMAGE_FILES:
                return error(404, "Image not included in this sample")
            return FileResponse(sample.directory / name, media_type="image/png")
        if v.startswith("sample-"):
            return error(410, "This saved sample is no longer available")
        if name not in image_names and name not in {"preview.png", "preview_hh.png", "preview_t1.png", "preview_t1_hh.png"}:
            return error(404, "Image not found")
        with demo.lock:
            if name in image_names and (demo.result_revision is None or v != str(demo.result_revision)):
                return error(410, "This result is no longer available")
            path = demo.dir / name
            if not path.is_file():
                return error(404, "Image not found")
            data = path.read_bytes()
        return Response(data, media_type="image/png")

    async def read_limited(request, limit):
        parts, total = [], 0
        async for part in request.stream():
            total += len(part)
            if total > limit:
                return None
            parts.append(part)
        return b"".join(parts)

    @app.post("/upload/{date}")
    async def upload(date: str, request: Request):
        if request.headers.get("x-nisar-demo") != "1":
            return error(403, "Open the demo page to submit requests.")
        if not model_enabled:
            return read_only()
        if date not in {"t0", "t1"}:
            return error(404, "Unknown endpoint.")
        data = await read_limited(request, MAX_CHUNK)
        if not data:
            return error(400, "Expected an upload chunk of at most 8 MiB. Refresh the demo page.")
        try:
            offset = int(request.headers.get("x-upload-offset", "-1"))
            total = int(request.headers.get("x-upload-size", "0"))
        except ValueError:
            return error(400, "Invalid upload headers.")
        return await run_in_threadpool(call_locked, demo, lambda: demo.upload(date, offset, total, data))

    @app.post("/prepare")
    def prepare(request: Request):
        if request.headers.get("x-nisar-demo") != "1":
            return error(403, "Open the demo page to submit requests.")
        if not model_enabled:
            return read_only()
        return call_locked(demo, demo.prepare)

    @app.post("/infer")
    async def infer(request: Request):
        if request.headers.get("x-nisar-demo") != "1":
            return error(403, "Open the demo page to submit requests.")
        if not model_enabled:
            return read_only()
        raw = await read_limited(request, 2048)
        if not raw:
            return error(400, "Expected a small selection JSON body.")
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("Expected a selection object.")
        except (ValueError, UnicodeDecodeError) as exc:
            return error(400, str(exc))
        with job_lock:
            if any(job["status"] == "running" for job in jobs.values()):
                return error(409, "An analysis is already running. Wait for it to finish.")
            # Keep only recent job statuses; the latest completed report also lives at /result.
            while len(jobs) >= 8:
                del jobs[next(iter(jobs))]
            run_id = uuid4().hex
            box = body.get("box")
            pixels = ((box[2]-box[0])*(box[3]-box[1])
                      if isinstance(box,list) and len(box)==4 and all(type(v) is int for v in box) else 1)
            jobs[run_id] = dict(run_id=run_id, status="running", stage="read",
                                message="Waiting for the model…", completed=None, total=None,
                                started=monotonic(), timing=RunTiming(pixels, timing_history))
            snapshot = job_snapshot(jobs[run_id])
        task = asyncio.create_task(run_job(run_id, body))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return JSONResponse(snapshot, status_code=202)

    @app.get("/infer/{run_id}")
    async def infer_progress(run_id: str):
        with job_lock:
            job = jobs.get(run_id)
            if job is None:
                return error(404, "This analysis is no longer available. The server may have restarted.")
            return job_snapshot(job)

    return app
