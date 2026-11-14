"""Map public browser URLs to feature-owned files through an explicit allowlist."""
from pathlib import Path

from fastapi.responses import FileResponse, JSONResponse


PAGES = {
    "/": "app/index.html",
    "/change": "features/change_viewer/change.html",
    "/viewer": "features/change_viewer/viewer.html",
}
STATIC_FILES = {
    "zoom.js": ("shared/map/zoom.js", "text/javascript"),
    "explorer.js": ("shared/map/explorer.js", "text/javascript"),
    "navigation.css": ("shared/navigation/navigation.css", "text/css"),
    "pick.js": ("features/selection/pick.js", "text/javascript"),
    "sample.js": ("features/samples/sample.js", "text/javascript"),
    "pick.css": ("features/selection/pick.css", "text/css"),
    "change.js": ("features/change_viewer/change.js", "text/javascript"),
    "change.css": ("features/change_viewer/change.css", "text/css"),
    "results.js": ("features/results/results.js", "text/javascript"),
    "results.css": ("features/results/results.css", "text/css"),
    "results-evidence.js": ("features/results/results-evidence.js", "text/javascript"),
    "report-view.js": ("features/reporting/report-view.js", "text/javascript"),
}
ICON_FILES = frozenset({"zoom-in", "zoom-out", "recenter", "fullscreen", "exit-fullscreen",
                        "split-handle", "swatch", "swatch-selected", "dot-t0", "dot-t1"})


def register_web_routes(app, root: Path):
    """Serve the browser without exposing source code, input data, or model files."""
    def file_route(relative_path, media_type):
        def response():
            return FileResponse(root / relative_path, media_type=media_type)
        return response

    for url, relative_path in PAGES.items():
        app.add_api_route(url, file_route(relative_path, "text/html"), methods=["GET"])

    for filename, (relative_path, media_type) in STATIC_FILES.items():
        app.add_api_route("/" + filename, file_route(relative_path, media_type),
                          methods=["GET"], include_in_schema=False)

    @app.get("/icons/{name}.svg", include_in_schema=False)
    def icon(name: str):
        if name not in ICON_FILES:
            return JSONResponse({"error": "Icon not found"}, status_code=404)
        return FileResponse(root / "shared" / "icons" / (name + ".svg"), media_type="image/svg+xml")
