"""Radar display tones and geographic alignment for image previews."""
import math

import numpy as np
from PIL import Image


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


def aligned_power(grid, box, shift, step):
    """Sample another date on the T0 pixel grid, keeping uncovered samples empty."""
    x, y, right, bottom = box
    dc, dr = shift
    height, width = math.ceil((bottom-y)/step), math.ceil((right-x)/step)
    power = np.zeros((2, height, width), np.float32)
    valid = np.zeros((height, width), bool)
    # Choose sample indices on this window's existing lattice, not a new origin
    # at the overlap boundary (which would shift subsampled T1 previews).
    c0, r0 = max(0, math.ceil((-dc-x)/step)), max(0, math.ceil((-dr-y)/step))
    c1 = min(width, math.ceil((grid.width-dc-x)/step))
    r1 = min(height, math.ceil((grid.height-dr-y)/step))
    if c0 < c1 and r0 < r1:
        sampled, usable = grid.power(x+c0*step+dc, y+r0*step+dr,
            min(right+dc, grid.width), min(bottom+dr, grid.height), step)
        power[:,r0:r1,c0:c1] = sampled
        valid[r0:r1,c0:c1] = usable
    return power, valid


