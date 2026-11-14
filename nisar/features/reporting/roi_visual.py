"""Visual evidence for the HDF5 ROI report."""
from io import BytesIO

import numpy as np
from PIL import Image



QWEN_PATCH = 16
QWEN_MERGE = 2
QWEN_ALIGNMENT = QWEN_PATCH * QWEN_MERGE
PANEL_NAMES = (
    "Clean T0 — earlier acquisition; HH/HV radar only",
    "Clean T1 — later acquisition; HH/HV radar only",
    "Standalone change map — white: flagged; black: valid and unflagged; gray: no paired data",
)


def native_views(z0,z1,v0,v1,limits,max_edge):
    """Render the whole selection, averaging all valid source pixels when shrinking."""
    h,w=v0.shape
    scale=min(1,max_edge/max(w,h))
    ow,oh=max(1,round(w*scale)),max(1,round(h*scale))
    xs=np.arange(ow+1,dtype=np.int64)*w//ow
    ys=np.arange(oh+1,dtype=np.int64)*h//oh
    images=[np.zeros((oh,ow,3),np.uint8) for _ in range(2)]
    lo,hi=np.asarray(limits,np.float32).T
    for row,(y0,y1) in enumerate(zip(ys[:-1],ys[1:],strict=True)):
        s=slice(y0,y1)
        for image,z,v in zip(images,(z0,z1),(v0,v1),strict=True):
            valid=v[s]
            counts=np.add.reduceat(valid.sum(axis=0),xs[:-1])
            sums=np.add.reduceat(np.where(valid[None],z[:,s],0).sum(axis=1),xs[:-1],axis=1)
            q=np.clip((sums/np.maximum(1,counts)-lo[:,None])/(hi-lo)[:,None],0,1)
            image[row,:,:2]=np.rint(np.where(counts,q,0).T*255).astype(np.uint8)
    # Preserve the ROI aspect ratio; add only the padding needed by the vision grid.
    def aligned(n):
        return (n+QWEN_ALIGNMENT-1)//QWEN_ALIGNMENT*QWEN_ALIGNMENT
    size=(aligned(ow),aligned(oh))
    result=[]
    for image in images:
        panel=Image.new("RGB",size)
        panel.paste(Image.fromarray(image),(0,0))
        result.append(panel)
    return result,(xs,ys)


def native_change_map(size, edges, mask, valid):
    """Use the same source bins and padding as the clean radar views."""
    xs,ys=edges
    panel=np.full((size[1],size[0]),128,dtype=np.uint8)
    for row,(y0,y1) in enumerate(zip(ys[:-1],ys[1:],strict=True)):
        good=valid[y0:y1]
        covered=np.logical_or.reduceat(good.any(axis=0),xs[:-1])
        flagged=np.logical_or.reduceat((mask[y0:y1]&good).any(axis=0),xs[:-1])
        panel[row,:len(covered)]=np.where(flagged,255,np.where(covered,0,128))
    return Image.fromarray(panel).convert("RGB")


def panel_grid(pictures):
    """Expected processor grids and merged visual token counts in panel order."""
    grids,counts=[],[]
    for picture in pictures:
        w,h=picture.size
        if w%QWEN_ALIGNMENT or h%QWEN_ALIGNMENT:
            raise ValueError("Visual panel dimensions must be multiples of 32.")
        grids.append([1,h//QWEN_PATCH,w//QWEN_PATCH])
        counts.append((w//QWEN_ALIGNMENT)*(h//QWEN_ALIGNMENT))
    return grids,counts


class DecoderTrace:
    """One result's visual inputs, captured at the Qwen call boundary."""
    def __init__(self, export_tensors=False):
        self.export_tensors=export_tensors
        self.panels=[]
        self.attempts=[]
        self.streams=[]
        self.pngs=[]
        self.tensor_archive=None
        self.windows=[]

    def record_panels(self,pictures):
        _,counts=panel_grid(pictures)
        start=0
        for i,(picture,count) in enumerate(zip(pictures,counts,strict=True)):
            with BytesIO() as out:
                picture.save(out,format="PNG")
                self.pngs.append(out.getvalue())
            group,role=divmod(i,len(PANEL_NAMES))
            self.panels.append(dict(slot=i,group=group,label=PANEL_NAMES[role],
                                    pixels=list(picture.size),encoding="qwen_vision",
                                    token_rows=[start,start+count]))
            start+=count

    def record_streams(self,main,detail):
        self.attempts[-1]["visual_tokens_observed"]=True
        if self.streams:
            return  # Keep the first capture of this request.
        streams=dict(main=main,**{f"deepstack_{i}":x for i,x in enumerate(detail)})
        self.streams=[dict(name=k,shape=list(v.shape),dtype=str(v.dtype)) for k,v in streams.items()]
        if self.export_tensors:
            arrays={}
            for name,tensor in streams.items():
                cpu=tensor.detach().cpu()
                # NumPy has no built-in bfloat16. Conversion to float32 is exact.
                arrays[name]=(cpu.float() if str(cpu.dtype)=="torch.bfloat16" else cpu).numpy()
            with BytesIO() as out:
                np.savez(out,**arrays)
                self.tensor_archive=out.getvalue()

    def manifest(self,revision):
        return dict(panels=[dict(p,url=f"/decoder-panel.png?v={revision}&i={p['slot']}") for p in self.panels],
                    windows=self.windows,attempts=self.attempts,streams=self.streams,
                    tensor_url=f"/decoder-tokens.npz?v={revision}" if self.tensor_archive is not None else None)


