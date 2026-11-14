"""Windowed reads from regular north-up NISAR L-SAR GCOV grids."""
from dataclasses import dataclass
import math
import re

import h5py
import numpy as np

CHANNELS = ("HHHH", "HVHV")
IDENT = "/science/LSAR/identification"


def scalar(group, name):
    ds = group[name]
    if not isinstance(ds, h5py.Dataset) or ds.size != 1:
        raise ValueError(f"Expected a scalar for {name}")
    return np.asarray(ds[()]).item()


@dataclass
class Grid:
    group: h5py.Group
    width: int
    height: int
    west: float
    north: float
    dx: float
    dy: float  # positive magnitude
    epsg: int

    @classmethod
    def read(cls, h, freq):
        g = h[f"/science/LSAR/GCOV/grids/frequency{freq}"]
        a, b = (g[k] for k in CHANNELS)
        if a.ndim != 2 or b.shape != a.shape or min(a.shape) < 1 or any(d.dtype.kind != "f" for d in (a,b)):
            raise ValueError("Expected two matching 2D real HHHH and HVHV power arrays.")
        height, width = a.shape
        dx, dy = float(scalar(g, "xCoordinateSpacing")), -float(scalar(g, "yCoordinateSpacing"))
        if not (np.isfinite([dx,dy]).all() and dx > 0 and dy > 0):
            raise ValueError("Expected a north-up grid with positive X and negative Y spacing.")
        x, y = (np.asarray(g[k][:], dtype=np.float64) for k in ("xCoordinates", "yCoordinates"))
        if (x.shape != (width,) or y.shape != (height,) or not np.isfinite(x).all()
                or not np.isfinite(y).all() or not np.allclose(np.diff(x),dx,rtol=0,atol=1e-5)
                or not np.allclose(np.diff(y),-dy,rtol=0,atol=1e-5)):
            raise ValueError("Invalid or irregular GCOV raster coordinates.")
        epsg = int(scalar(g, "projection"))
        if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760 or epsg in (3413,3031)):
            raise ValueError("This demo expects a supported metric UTM or polar grid.")
        return cls(g,width,height,float(x[0]-dx/2),float(y[0]+dy/2),dx,dy,epsg)

    def power(self, x0, y0, x1, y1, step=1):
        if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
            raise ValueError("Read outside the GCOV grid.")
        out, valid = [], None
        for name in CHANNELS:
            d = self.group[name]
            p = np.asarray(d[y0:y1:step,x0:x1:step], dtype=np.float32)
            ok = np.isfinite(p) & (p > 0)
            for fill in [d.fillvalue, *np.ravel(d.attrs.get("_FillValue", [])),
                         *np.ravel(d.attrs.get("missing_value", []))]:
                if fill is not None and np.isfinite(fill):
                    ok &= p != fill
            out.append(p)
            valid = ok if valid is None else valid & ok
        for p in out:
            p[~valid] = 0
        return np.stack(out), valid

    def rtc_median(self, box, valid):
        """Optional co-gridded GCOV gamma-to-sigma factor; no radiometric conversion."""
        d = self.group.get("rtcGammaToSigmaFactor")
        if not isinstance(d,h5py.Dataset) or d.shape != (self.height,self.width) or d.dtype.kind != "f":
            return None
        x0,y0,x1,y1 = box
        q = d[y0:y1,x0:x1]
        ok = valid & np.isfinite(q) & (q > 0)
        for fill in [d.fillvalue, *np.ravel(d.attrs.get("_FillValue", [])),
                     *np.ravel(d.attrs.get("missing_value", []))]:
            if fill is not None and np.isfinite(fill):
                ok &= q != fill
        return round(float(np.median(q[ok])),6) if ok.any() else None

    def bounds(self, box):
        x0,y0,x1,y1 = box
        return [self.west+x0*self.dx,self.north-y1*self.dy,
                self.west+x1*self.dx,self.north-y0*self.dy]


def match(a,b):
    if a.epsg != b.epsg or not np.allclose([a.dx,a.dy],[b.dx,b.dy],rtol=0,atol=1e-6):
        raise ValueError("Different CRS or native pixel spacing. Pre-align the pair; this demo does not warp it.")
    exact = [(a.west-b.west)/b.dx,(b.north-a.north)/b.dy]
    shift = tuple(round(v) for v in exact)
    if not np.allclose(exact,shift,rtol=0,atol=1e-5):
        raise ValueError("Fractional-pixel grid shift. Pre-align the pair; rounding would compare different locations.")
    dc,dr = shift
    overlap = [max(0,-dc),max(0,-dr),min(a.width,b.width-dc),min(a.height,b.height-dr)]
    if overlap[0]>=overlap[2] or overlap[1]>=overlap[3]:
        raise ValueError("The scenes have no common grid footprint.")
    return shift,overlap


def move(box, shift):
    x,y = shift
    return [box[0]+x,box[1]+y,box[2]+x,box[3]+y]


def roi_box(box, a, overlap, limit=None):
    if not isinstance(box,list) or len(box)!=4 or any(type(x) is not int for x in box):
        raise ValueError("Selection must contain four integer pixel coordinates.")
    x0,y0,x1,y1 = box
    if not (0 <= x0 < x1 <= a.width and 0 <= y0 < y1 <= a.height):
        raise ValueError("Selection is outside T0 or empty.")
    if not (overlap[0]<=x0<x1<=overlap[2] and overlap[1]<=y0<y1<=overlap[3]):
        raise ValueError("Select inside the outlined common grid footprint.")
    if limit is not None and (x1-x0)*(y1-y0)>limit:
        raise ValueError(f"Select at most {limit:,} native pixels for this local demo.")
    return box


def db(p):
    out = np.full(p.shape, np.nan, dtype=np.float32)
    np.log10(p, out=out, where=p>0)
    out *= 10
    return out


def factor(grid, spacing):
    f = [spacing/grid.dy,spacing/grid.dx]
    if any(not math.isfinite(v) or v<1 or not math.isclose(v,round(v),rel_tol=0,abs_tol=1e-6) for v in f):
        raise ValueError("Checkpoint spacing must be an integer multiple of native spacing; no arbitrary resampling.")
    return tuple(round(v) for v in f)


def to_model(p, valid, fy, fx):
    _, h, w = p.shape
    hh, ww = math.ceil(h/fy), math.ceil(w/fx)
    pad = ((0,hh*fy-h),(0,ww*fx-w))
    if fy == fx == 1:
        return db(p),valid
    p = np.pad(p, ((0,0),*pad))
    v = np.pad(valid, pad).reshape(hh,fy,ww,fx).all((1,3))
    q = p.reshape(2,hh,fy,ww,fx).mean((2,4), dtype=np.float64).astype(np.float32)
    q[:,~v] = 0
    return db(q),v


def metadata(h):
    g = h.get(IDENT)
    if g is None:
        return {}
    result = {}
    for k in ("zeroDopplerStartTime","zeroDopplerEndTime","orbitPassDirection","lookDirection",
              "trackNumber","frameNumber","absoluteOrbitNumber","relativeOrbitNumber"):
        if k not in g:
            continue
        v = scalar(g,k)
        if isinstance(v,bytes):
            v = v.decode("ascii",errors="replace").strip()
        if k.endswith("Time") and isinstance(v,str) and re.fullmatch(r"[0-9TZ:.+ -]{10,48}",v):
            result[k] = v
        elif k.endswith("Direction") and isinstance(v,str) and v.lower() in ("ascending","descending","left","right"):
            result[k] = v.lower()
        elif type(v) is int:
            result[k] = v
    return result
