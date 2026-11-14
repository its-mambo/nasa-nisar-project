"""FastAPI routes for the NISAR analysis demo."""

from io import BytesIO
import json
from pathlib import Path
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool


MAX_CHUNK = 8 * 1024 * 1024
STATIC_FILES = {"index.html": "text/html", "viewer.html": "text/html",
                "zoom.js": "text/javascript", "pick.js": "text/javascript",
                "results.js": "text/javascript", "pick.css": "text/css",
                "results.css": "text/css"}


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


def create_app(demo, root: Path, names):
    app = FastAPI(title="NISAR SAR Change Explorer")
    image_names = set(names)

    @app.middleware("http")
    async def no_cache(request, next_call):
        response = await next_call(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(root / "index.html", media_type="text/html")

    @app.get("/viewer")
    def viewer():
        return FileResponse(root / "viewer.html", media_type="text/html")

    for filename, media_type in STATIC_FILES.items():
        if filename in {"index.html", "viewer.html"}:
            continue
        def static_route(name, content_type):
            def file_response():
                return FileResponse(root / name, media_type=content_type)
            return file_response
        app.add_api_route("/" + filename, static_route(filename, media_type),
                          methods=["GET"], include_in_schema=False)

    @app.get("/pair")
    def pair():
        with demo.lock:
            return {"pair": demo.info}

    @app.get("/decoder-inputs")
    def decoder_inputs(v: str = ""):
        with demo.lock:
            revision, trace = demo.result_revision, demo.decoder_trace
        if revision is None or v != str(revision):
            return error(410, "This result is no longer available")
        if trace is None:
            return error(404, "No visual decoder inputs for this result")
        return trace.manifest(revision)

    @app.get("/decoder-tokens.npz")
    def decoder_tokens(v: str = ""):
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
    def preview_tile(v: str = "", mode: str = "", step: str = "", col: str = "", row: str = ""):
        def render():
            if demo.info is None or v != str(demo.info["revision"]):
                return error(410, "This preview is no longer available")
            tile = demo.preview_tile(mode, int(step), int(col), int(row))
            with BytesIO() as output:
                tile.save(output, format="PNG", compress_level=1)
                return Response(output.getvalue(), media_type="image/png")
        return call_locked(demo, render)

    @app.get("/{filename}.png")
    def png(filename: str, v: str = ""):
        name = filename + ".png"
        if name not in image_names and name not in {"preview.png", "preview_hh.png"}:
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
        return call_locked(demo, demo.prepare)

    @app.post("/infer")
    async def infer(request: Request):
        if request.headers.get("x-nisar-demo") != "1":
            return error(403, "Open the demo page to submit requests.")
        raw = await read_limited(request, 2048)
        if not raw:
            return error(400, "Expected a small selection JSON body.")
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("Expected a selection object.")
        except (ValueError, UnicodeDecodeError) as exc:
            return error(400, str(exc))
        return await run_in_threadpool(call_locked, demo,
                                       lambda: demo.infer(body.get("box"), body.get("revision")))

    return app
