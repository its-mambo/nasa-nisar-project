"""SAR detector and trained scene classifier from the NISAR checkpoint."""

from contextlib import nullcontext
import json

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

from torch.utils.checkpoint import checkpoint

TILE_SIZE = 224

BUNDLE_FORMAT = "nisar_inference_weights_v1"

class PatchEmbed(nn.Module):
    def __init__(self,dim,patch_size):
        super().__init__();self.proj=nn.Conv2d(2,dim,patch_size,stride=patch_size)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)

class Attention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();self.heads=heads
        self.qkv=nn.Linear(dim,3*dim);self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        b,n,d=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        return self.proj(y.transpose(1,2).reshape(b,n,d))

class MLP(nn.Module):
    def __init__(self,dim):
        super().__init__();self.fc1=nn.Linear(dim,4*dim);self.fc2=nn.Linear(4*dim,dim)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();self.norm1=nn.LayerNorm(dim,eps=1e-6);self.norm2=nn.LayerNorm(dim,eps=1e-6)
        self.attn=Attention(dim,heads);self.mlp=MLP(dim)
    def forward(self,x):
        x=x+self.attn(self.norm1(x));return x+self.mlp(self.norm2(x))

class Encoder(nn.Module):
    def __init__(self,size,arch,grad_checkpoint=True):
        super().__init__()
        dim,depth,heads={'huge':(1280,32,16),'tiny':(64,2,4)}[arch]
        if size%14: raise ValueError('Input size must be divisible by patch size 14')
        self.image_size=size;self.patch_size=14;self.dim=dim;self.grad_checkpoint=grad_checkpoint
        self.patch_embed=PatchEmbed(dim,14)
        self.cls_token=nn.Parameter(torch.zeros(1,1,dim))
        self.pos_embed=nn.Parameter(torch.zeros(1,(size//14)**2+1,dim),requires_grad=False)
        self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim,eps=1e-6)

    def forward(self,x):
        if x.shape[1:]!=(2,self.image_size,self.image_size): raise ValueError('Input/checkpoint tile-size mismatch')
        z=self.patch_embed(x)+self.pos_embed[:,1:]
        cls=(self.cls_token+self.pos_embed[:,:1]).expand(x.shape[0],-1,-1)
        z=torch.cat((cls,z),1)
        for block in self.blocks:
            if self.training and self.grad_checkpoint and any(p.requires_grad for p in block.parameters()):
                z=checkpoint(block,z,use_reentrant=False)
            else: z=block(z)
        z=self.norm(z)[:,1:]
        side=self.image_size//14
        return z.transpose(1,2).reshape(x.shape[0],self.dim,side,side)

def conv(cin,cout):
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(8,cout),nn.GELU())

class ChangeModel(nn.Module):
    def __init__(self,size,arch='huge',width=128,grad_checkpoint=True):
        super().__init__()
        if width<16 or width%8: raise ValueError('Head width must be >=16 and divisible by 8')
        self.encoder=Encoder(size,arch,grad_checkpoint)
        # Mean and absolute difference make prediction invariant to temporal order.
        self.fuse=nn.Sequential(nn.Conv2d(self.encoder.dim*2,width,1),nn.GroupNorm(8,width),nn.GELU(),conv(width,width))
        self.local=nn.Sequential(conv(4,32),conv(32,32))
        self.refine=nn.Sequential(conv(width+32,64),conv(64,32),nn.Conv2d(32,1,1))

    def forward(self,a,b):
        f0=self.encoder(a);f1=self.encoder(b)
        z=self.fuse(torch.cat(((f0+f1)*.5,(f1-f0).abs()),1))
        z=F.interpolate(z,size=a.shape[-2:],mode='bilinear',align_corners=False)
        skip=self.local(torch.cat(((a+b)*.5,(b-a).abs()),1))
        return self.refine(torch.cat((z,skip),1))[:,0]

    def set_trainable(self,last_blocks):
        depth=len(self.encoder.blocks)
        if not 0<=last_blocks<=depth: raise ValueError(f'train-blocks must be in [0,{depth}]')
        for p in self.encoder.parameters(): p.requires_grad_(False)
        if last_blocks:
            for block in self.encoder.blocks[-last_blocks:]:
                for p in block.parameters(): p.requires_grad_(True)
            for p in self.encoder.norm.parameters(): p.requires_grad_(True)

def load_payload(path):
    if str(path).endswith(".safetensors"):
        return load_sar(path)
    return torch.load(path,map_location='cpu',weights_only=True,mmap=True)

SAR_SAFETENSORS_FORMAT = "nisar_sar_safetensors_v1"


def save_sar(payload, path):
    from safetensors.torch import save_file

    detector = dict(payload["detector"])
    linear = dict(payload["linear"])
    tensors = {}
    for prefix, state in (("detector.model", detector.pop("model")),
                          ("linear.model", linear.pop("model"))):
        tensors.update({f"{prefix}.{k}": v for k, v in state.items()})
    for name in ("center", "scale"):
        tensors[f"linear.{name}"] = linear.pop(name)

    metadata = dict(detector=detector, linear=linear,
                    detector_sha256=payload["detector_sha256"])
    save_file({name: t.detach().cpu().contiguous() for name, t in tensors.items()},
              str(path), metadata={"format": SAR_SAFETENSORS_FORMAT,
                                   "config": json.dumps(metadata, allow_nan=False)})


def load_sar(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        header = f.metadata() or {}
        if header.get("format") != SAR_SAFETENSORS_FORMAT:
            raise ValueError("Expected a NISAR SAR safetensors file.")
        payload = json.loads(header["config"])
        keys = set(f.keys())
        for group in ("detector", "linear"):
            prefix = f"{group}.model."
            names = {k for k in keys if k.startswith(prefix)}
            if not names:
                raise ValueError(f"Missing {group} state in SAR safetensors.")
            payload[group]["model"] = {k[len(prefix):]: f.get_tensor(k) for k in names}
            keys -= names
        if keys != {"linear.center", "linear.scale"}:
            raise ValueError("Invalid classifier normalization in SAR safetensors.")
        for name in ("center", "scale"):
            payload["linear"][name] = f.get_tensor(f"linear.{name}")
    return payload


SLOTS = 24

def pool_features(f0, f1, region, valid, patch_size=14):
    def pooled(z, mask):
        w = F.avg_pool2d(mask[None, None].float(), patch_size, patch_size)
        if w.shape[-2:] != z.shape[-2:]:
            raise ValueError("Mask/encoder patch grid mismatch")
        return (z.float() * w).sum((-2, -1))[0] / w.sum().clamp_min(1e-12)

    p0, p1 = pooled(f0, region), pooled(f1, region)
    ctx = valid & ~region
    c0, c1 = pooled(f0, ctx), pooled(f1, ctx)
    return torch.stack((p0, p1, p1 - p0, c0, c1, c1 - c0))

def choose_dtype(name, device):
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]

def verify_run(linear, sha, encoder_width):
    labels=linear["classes"]
    if (not isinstance(labels,(list,tuple)) or len(labels)<2
            or any(not isinstance(label,str) or not label.strip() for label in labels)
            or len(set(labels))!=len(labels)):
        raise ValueError("Classifier checkpoint must contain ordered, unique class names.")
    if linear["encoder_sha256"] != sha:
        raise ValueError("Linear checkpoint SAR checkpoint hash does not match.")
    width=SLOTS*encoder_width
    # The scene trainer saves one scalar RMS scale; per-feature scales also broadcast.
    if (linear["center"].shape != (width,) or linear["scale"].shape not in ((),(1,),(width,))
            or linear["model"]["weight"].shape != (len(labels),width)
            or linear["model"]["bias"].shape != (len(labels),)):
        raise ValueError("Classifier checkpoint dimensions do not match its labels and encoder features.")


def build_prefix(encoder, tiles, size, mean, std, clip, device, dtype, min_pixels, min_valid):
    try:
        from scipy import ndimage as ndi
    except ImportError as exc:
        raise ValueError("Scene classification requires scipy (pip install scipy).") from exc

    encoder.to(dtype=dtype)
    crop_count = region_count = 0
    context_sum = None
    best = []
    with torch.inference_mode():
        for row, col, db0, db1, usable, selected in tiles:
            if usable.sum() < min_valid * size**2:
                continue
            h, w = usable.shape
            valid = np.zeros((size, size), bool)
            valid[:h, :w] = usable
            change = np.zeros_like(valid)
            change[:h, :w] = selected & usable
            cc, _ = ndi.label(change, structure=np.ones((3, 3), np.uint8))
            counts = np.bincount(cc.ravel())
            regions = [(k, box) for k, box in enumerate(ndi.find_objects(cc), 1)
                       if counts[k] >= min_pixels]
            if not regions:
                continue

            def encode(db):
                x = np.zeros((2, size, size), np.float32)
                db = np.clip(db, clip[:, 0, None, None], clip[:, 1, None, None])
                x[:, :h, :w] = np.where(usable[None], (db - mean) / std, 0.0)
                return encoder(torch.from_numpy(x)[None].to(device=device, dtype=dtype))

            f0, f1 = encode(db0), encode(db1)
            valid_t = torch.from_numpy(valid).to(device)
            crop_sum = None
            rep = None
            for k, box in regions:
                region = cc == k
                # The training exporter flags invalid pixels in the one-pixel cross
                # neighborhood; pixels beyond the tile itself are represented by the edge flag.
                clean = not (ndi.binary_dilation(region) & ~valid).any()
                edge = any(b.start == 0 or b.stop == size for b in box)
                rank = (edge, -int(counts[k]), row, col, k)
                v = pool_features(f0, f1, torch.from_numpy(region).to(device), valid_t,
                                  patch_size=encoder.patch_size)
                crop_sum = v.clone() if crop_sum is None else crop_sum.add_(v)
                if clean and (rep is None or rank < rep[0]):
                    rep = (rank, v)
            crop_mean = crop_sum / len(regions)
            context_sum = crop_mean if context_sum is None else context_sum.add_(crop_mean)
            crop_count += 1
            region_count += len(regions)
            if rep is not None:
                best.append(rep)
                best.sort(key=lambda item: item[0])
                del best[3:]
            if crop_count % 250 == 0:
                print(f"  Scene features: {crop_count:,} crops / {region_count:,} regions", flush=True)

        report = {
            "candidate_regions": region_count,
            "candidate_crops": crop_count,
            "representatives": [
                {"row": rank[2], "col": rank[3], "component": rank[4],
                 "pixels": -rank[1], "touches_tile_edge": bool(rank[0])}
                for rank, _ in best
            ],
            "min_region_pixels": min_pixels,
            "min_valid_fraction": min_valid,
        }
        if len(best) < 3:
            return None, report
        prefix = torch.cat([context_sum / crop_count, *(v for _, v in best)], dim=0).cpu()
        if prefix.shape != (SLOTS, encoder.dim) or not torch.isfinite(prefix).all():
            raise ValueError("Unexpected or non-finite scene features.")
        return prefix, report

def classify(prefix, report, linear):
    if prefix is None:
        return {**report, "status": "insufficient_candidates", "predictions": {},
                "reason": "Need eligible components away from invalid pixels in three distinct crops."}
    with torch.inference_mode():
        z = F.normalize((prefix.flatten()[None] - linear["center"]) / linear["scale"])
        logits = F.linear(z, linear["model"]["weight"], linear["model"]["bias"])
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite linear scene prediction.")
        scores = logits[0].softmax(-1)
        labels=linear["classes"]
        prediction = {"label": labels[int(scores.argmax())],
                      "scores": dict(zip(labels, scores.tolist(), strict=True))}
    return {**report, "status": "ok", "predictions": {"linear": prediction},
            "score_note": "Experimental softmax scores; not calibrated probabilities."}

def scene_dtype(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            return choose_dtype("auto", device)
    return choose_dtype("auto", device)

def ranges(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (2, 2) or not np.isfinite(value).all() or (value[:, 1] <= value[:, 0]).any():
        raise ValueError("Expected finite, increasing dB ranges for HH and HV.")
    return value

def checkpoint_spec(payload):
    if payload.get("format") != "nisar_change_v1" or payload.get("size") != TILE_SIZE:
        raise ValueError("Expected a fine-tuned nisar_change_v1 checkpoint for 224×224 tiles.")
    stats = payload["stats"]
    if tuple(stats["channels"]) != ("HHHH", "HVHV"):
        raise ValueError("Checkpoint must use HH, HV in that order.")
    clip = ranges([stats["db_clip"]] * 2)
    return payload, clip

def offsets(length, stride):
    # Keep the fixed grid and pad only the trailing tile when needed.
    return range(0, max(0, length - TILE_SIZE) + stride, stride)

def score_alpha(scores):
    alpha = np.rint(np.clip(scores, 0, 1) * 255).astype(np.uint8)
    alpha[(scores > 0) & (alpha == 0)] = 1
    return alpha

def infer_scene(model, size, read_pair, mean, std, clip, device, amp, batch_size, overlap,
                threshold, score_opacity=False):
    """Predict a stitched mask. read_pair(box) supplies ((dB0, valid0), (dB1, valid1))."""
    width, height = size
    stride = TILE_SIZE - overlap
    rows, cols = offsets(height, stride), offsets(width, stride)
    total = len(rows) * len(cols)
    print(f"Detector: {width}×{height} pixels, {total:,} tiles of {TILE_SIZE}×{TILE_SIZE}, "
          f"overlap {overlap}, {device}, AMP {amp}.", flush=True)
    # Rolling strip: float accumulators scale with width × tile size, not scene area.
    sums = np.zeros((TILE_SIZE, width), np.float32)
    weights = np.zeros_like(sums)
    change_mask = np.zeros((height, width), dtype=bool)
    alpha = np.zeros((height, width), np.uint8) if score_opacity else None
    taper = np.maximum(np.hanning(TILE_SIZE).astype(np.float32), .05) if overlap else np.ones(TILE_SIZE, np.float32)
    kernel = taper[:, None] * taper[None, :]
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    valid_count = changed_count = done = 0
    with torch.inference_mode():
        for row in rows:
            h = min(height - row, TILE_SIZE)
            for start in range(0, len(cols), batch_size):
                inputs = [[], []]
                places = []
                for col in cols[start:start + batch_size]:
                    w = min(width - col, TILE_SIZE)
                    box = (col, row, col + w, row + h)
                    x, y = read_pair(box)
                    valid = x[1] & y[1]
                    if not valid.any():
                        continue
                    for (db, _), items in zip((x, y), inputs, strict=True):
                        tile = np.zeros((2, TILE_SIZE, TILE_SIZE), np.float32)
                        db = np.clip(db, clip[:, 0, None, None], clip[:, 1, None, None])
                        tile[:, :h, :w] = np.where(valid[None], (db - mean) / std, 0.0)
                        items.append(tile)
                    places.append((col, w, valid))
                if places:
                    tensors = [torch.from_numpy(np.stack(items)).to(device) for items in inputs]
                    with torch.autocast("cuda", dtype=amp_dtype) if amp != "off" else nullcontext():
                        logits = model(*tensors)
                    batch = logits.float().sigmoid().cpu().numpy()
                    if batch.shape != (len(places), TILE_SIZE, TILE_SIZE):
                        raise ValueError(f"Unexpected detector output shape: {batch.shape}.")
                    for probs, (col, w, valid) in zip(batch, places, strict=True):
                        probs = probs[:h, :w]
                        if not np.isfinite(probs[valid]).all():
                            raise ValueError("Non-finite detector scores; try --amp off.")
                        win = np.s_[:h, col:col + w]
                        weight = kernel[:h, :w] * valid
                        sums[win] += np.where(valid, probs, 0.0) * weight
                        weights[win] += weight
                done += len(cols[start:start + batch_size])
                if done == total or done % 100 < batch_size:
                    print(f"  {done:,}/{total:,} tiles", flush=True)
            # Only emit rows that future tiles cannot touch.
            end = h if row == rows[-1] else stride
            covered = weights[:end] > 0
            scores = np.divide(sums[:end], weights[:end], out=np.zeros_like(sums[:end]), where=covered)
            changed = covered & (scores >= threshold)
            change_mask[row:row + end] = changed
            if alpha is not None:
                alpha[row:row + end] = score_alpha(scores)
            valid_count += int(covered.sum())
            changed_count += int(changed.sum())
            if overlap and row != rows[-1]:
                for strip in (sums, weights):
                    strip[:overlap] = strip[stride:].copy()
                    strip[overlap:] = 0
            else:
                sums.fill(0)
                weights.fill(0)
    return change_mask, alpha, valid_count, changed_count

def inference_device(device_name, amp):
    device = torch.device(device_name if device_name != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose an available CPU or CUDA device.")
    if amp == "auto":
        amp = "off"
        if device.type == "cuda":
            with torch.cuda.device(device):
                amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if amp != "off" and device.type != "cuda":
        raise ValueError("Use --amp off on CPU.")
    return device, amp

def build_scene_prefix(encoder, size, read_pair, mask, normalization, device, dtype, min_pixels, min_valid):
    """Build the trained scene representation on a fixed grid, even with detector overlap."""
    width, height = size

    def tiles():
        for row in range(0, height, TILE_SIZE):
            for col in range(0, width, TILE_SIZE):
                end_x, end_y = min(width, col + TILE_SIZE), min(height, row + TILE_SIZE)
                selected = mask[row:end_y, col:end_x]
                if selected.sum() < min_pixels:
                    continue
                (db0, ok0), (db1, ok1) = read_pair((col, row, end_x, end_y))
                yield row, col, db0, db1, ok0 & ok1, selected

    return build_prefix(encoder, tiles(), TILE_SIZE, *normalization, device, dtype, min_pixels, min_valid)
