"""Synthetic GCOV/model fixtures and an in-process ASGI request harness."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import h5py
import numpy as np

from nisar.features.analysis.service import Demo


class FakeSar:
    spec = {"frequency":"A", "spacing_m":10, "db_clip":[-30, 5]}
    device = SimpleNamespace(type="cpu")

    def detect(self, size, pair, *, threshold, progress, **kwargs):
        a, b = pair((0, 0, *size))
        valid = a[1] & b[1]
        mask = valid & (.6 >= threshold)
        progress(0, 1)
        progress(1, 1)
        return mask, None, int(valid.sum()), int(mask.sum())

def fixture_demo(directory, size=128):
    options = SimpleNamespace(threshold=.1, max_roi_pixels=1_000_000,
        batch_size=2, overlap=0, max_delta_db=8, max_upload_gib=1)
    demo = Demo(directory, options, FakeSar())
    yy, xx = np.mgrid[:size, :size]
    for date, scale, stamp in (("t0", 1., "2026-06-21T00:00:00Z"),
                                ("t1", 2., "2026-09-13T00:00:00Z")):
        with h5py.File(Path(directory) / f"{date}.h5", "w") as product:
            grid = product.create_group("/science/LSAR/GCOV/grids/frequencyA")
            grid["HHHH"] = ((.01 + xx / 5000 + yy / 10000) * scale).astype(np.float32)
            grid["HVHV"] = ((.002 + yy / 15000) * scale).astype(np.float32)
            grid["xCoordinateSpacing"], grid["yCoordinateSpacing"] = 10., -10.
            grid["xCoordinates"] = 500005. + np.arange(size)*10
            grid["yCoordinates"] = 4500005. - np.arange(size)*10
            grid["projection"] = 32640
            ident = product.create_group("/science/LSAR/identification")
            ident["zeroDopplerStartTime"] = stamp
    demo.prepare()
    return demo


async def request(app, method, path, body=None, authorized=True):
    """Exercise ASGI directly, without a network listener or httpx dependency."""
    parsed = urlsplit(path)
    payload = b"" if body is None else json.dumps(body).encode()
    sent, delivered = [], False
    headers = [(b"content-type", b"application/json")]
    if authorized:
        headers.append((b"x-nisar-demo", b"1"))
    scope = dict(type="http", asgi={"version":"3.0", "spec_version":"2.4"},
        method=method, path=parsed.path, raw_path=parsed.path.encode(),
        query_string=parsed.query.encode(), headers=headers, http_version="1.1",
        scheme="http", server=("test",80), client=("test",123))

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type":"http.request", "body":payload, "more_body":False}
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(app(scope, receive, send), 3)
    status = next(item["status"] for item in sent if item["type"] == "http.response.start")
    data = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
    return status, data


