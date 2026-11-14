"""Persistent SAR and Qwen models; Qwen explains evidence without changing SAR outputs."""
import base64
from collections import Counter
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import shutil
import struct
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from roi_visual import QWEN_PATCH, QWEN_MERGE, panel_grid

from qwen_evidence_prompt import messages as evidence_messages


FORMAT = "nisar_qwen_dual_v1"
DIRECTORY_FORMAT = "nisar_qwen_dual_safetensors_v1"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def manifest_members(manifest, *, directory=False):
    format_name, sar_file = ((DIRECTORY_FORMAT, "sar.safetensors") if directory
                             else (FORMAT, "sar.pt"))
    if (not isinstance(manifest, dict) or manifest.get("format") != format_name
            or manifest.get("sar_file") != sar_file):
        raise ValueError(f"Expected a {format_name} weights layout.")
    names = manifest.get("qwen_files")
    if not isinstance(names,list) or not names or any(
            not isinstance(n,str) or n in ("",".","..") or any(c in n for c in "/\\:\0") for n in names):
        raise ValueError("Invalid Qwen member names.")
    adapter = manifest.get("qwen_adapter_files", [])
    if adapter not in ([],list(ADAPTER_FILES)):
        raise ValueError("Invalid Qwen adapter member list.")
    members = [sar_file, *(f"qwen/{n}" for n in names), *(f"qwen_adapter/{n}" for n in adapter)]
    if len(set(members)) != len(members):
        raise ValueError("Weights manifest has duplicate members.")
    if directory and any(n.lower().endswith((".bin", ".pt", ".pth", ".ckpt")) for n in names):
        raise ValueError("The safetensors layout must not contain PyTorch weights.")
    return members


def bundle_members(z):
    if z.namelist().count("manifest.json") != 1:
        raise ValueError("Expected exactly one manifest.json.")
    manifest = json.loads(z.read("manifest.json"))
    members = manifest_members(manifest)
    expected = Counter(["manifest.json", *members])
    if Counter(z.namelist()) != expected:
        raise ValueError("Weights archive has missing, duplicate or unexpected members.")
    return manifest,members


SINGLE_FORMAT = "nisar_single_safetensors_v1"
HEADER_LIMIT = 100_000_000
CHUNK = 8 * 1024**2


def read_header(f):
    raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("Truncated safetensors file.")
    length, = struct.unpack("<Q", raw)
    if not 2 <= length <= HEADER_LIMIT:
        raise ValueError("Invalid safetensors header size.")
    raw = f.read(length)
    if len(raw) != length or not raw.startswith(b"{"):
        raise ValueError("Invalid safetensors header.")
    header = json.loads(raw)
    start = f.tell()
    end = f.seek(0, 2)
    cursor = 0
    spans = sorted(v["data_offsets"] for k, v in header.items() if k != "__metadata__")
    for lo, hi in spans:
        if lo != cursor or hi < lo:
            raise ValueError("Invalid or overlapping safetensors offsets.")
        cursor = hi
    if cursor != end - start:
        raise ValueError("Safetensors data length does not match its header.")
    return header, start, cursor


def write_header(f, header):
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    if len(raw) > HEADER_LIMIT:
        raise ValueError("Combined header exceeds the safetensors size limit.")
    f.write(struct.pack("<Q", len(raw)))
    f.write(raw)


def copy_range(src, dst, start, size):
    src.seek(start)
    while size:
        block = src.read(min(size, CHUNK))
        if not block:
            raise ValueError("Truncated tensor data.")
        dst.write(block)
        size -= len(block)


def pack_directory(root, out):
    """Write one standard safetensors container with namespaced tensor keys."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    members = manifest_members(manifest, directory=True)
    tensors, configs, parts, sources = {}, {}, [], []
    cursor = 0
    for name in members:
        path = root / name
        if not name.endswith(".safetensors"):
            configs[name] = base64.b64encode(path.read_bytes()).decode("ascii")
            continue
        with path.open("rb") as f:
            header, start, size = read_header(f)
        prefix = f"part{len(parts)}."
        parts.append(dict(name=name, metadata=header.pop("__metadata__", None),
                          offset=cursor, size=size))
        for key, item in header.items():
            lo, hi = item["data_offsets"]
            tensors[prefix + key] = dict(item, data_offsets=[cursor + lo, cursor + hi])
        sources.append((path, start, size))
        cursor += size
    tensors["__metadata__"] = {
        "format": SINGLE_FORMAT,
        "bundle": json.dumps(dict(manifest=manifest, configs=configs, parts=parts),
                             separators=(",", ":")),
    }
    with out.open("xb") as f:
        try:
            write_header(f, tensors)
            for path, start, size in sources:
                with path.open("rb") as src:
                    copy_range(src, f, start, size)
        except BaseException:
            f.close()
            out.unlink()
            raise


def restore_weights(path, root):
    """Restore the original component layout in a fresh runtime directory."""
    with path.open("rb") as src:
        header, start, total = read_header(src)
        meta = header.pop("__metadata__", {})
        if meta.get("format") != SINGLE_FORMAT:
            raise ValueError("Expected a combined NISAR safetensors file.")
        bundle = json.loads(meta["bundle"])
        manifest = bundle["manifest"]
        configs, parts = bundle["configs"], bundle["parts"]
        expected = set(manifest_members(manifest, directory=True))
        names = [*configs, *(p["name"] for p in parts)]
        if set(names) != expected or len(names) != len(expected) or any(
                n.endswith(".safetensors") for n in configs) or any(
                not p["name"].endswith(".safetensors") for p in parts):
            raise ValueError("Invalid combined weights manifest.")

        payloads = [dict() for _ in parts]
        for key, item in header.items():
            prefix, dot, tensor_name = key.partition(".")
            index = int(prefix[4:])
            if not dot or not tensor_name or not 0 <= index < len(parts) or prefix != f"part{index}":
                raise ValueError("Unknown tensor component.")
            part = parts[index]
            lo, hi = (v - part["offset"] for v in item["data_offsets"])
            if not 0 <= lo <= hi <= part["size"] or tensor_name == "__metadata__":
                raise ValueError("Tensor extends outside its component.")
            payloads[index][tensor_name] = dict(item, data_offsets=[lo, hi])

        cursor = 0
        for part, payload in zip(parts, payloads, strict=True):
            covered = 0
            for lo, hi in sorted(t["data_offsets"] for t in payload.values()):
                if lo != covered:
                    raise ValueError("Component has missing tensor data.")
                covered = hi
            if part["offset"] != cursor or part["size"] != covered or not payload:
                raise ValueError("Invalid component boundaries.")
            cursor += covered
        if cursor != total:
            raise ValueError("Combined weights contain unassigned tensor data.")

        decoded = {name: base64.b64decode(value, validate=True) for name, value in configs.items()}
        if shutil.disk_usage(root).free < total + start + sum(map(len, decoded.values())):
            raise OSError("Not enough space to stage weights; set --work-dir to a larger drive.")
        for name, raw in decoded.items():
            dest = root / name
            dest.parent.mkdir(exist_ok=True)
            with dest.open("xb") as f:
                f.write(raw)
        for part, payload in zip(parts, payloads, strict=True):
            if part["metadata"] is not None:
                payload["__metadata__"] = part["metadata"]
            dest = root / part["name"]
            dest.parent.mkdir(exist_ok=True)
            with dest.open("xb") as f:
                write_header(f, payload)
                copy_range(src, f, start + part["offset"], part["size"])
        with (root / "manifest.json").open("x", encoding="utf-8") as f:
            json.dump(manifest, f)


def convert_weights(source, out):
    from change_model import checkpoint_spec, load_payload, save_sar, verify_run

    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {out}")
    with ZipFile(source) as old:
        manifest, members = bundle_members(old)
        manifest.update(format=DIRECTORY_FORMAT, sar_file="sar.safetensors")
        manifest_members(manifest, directory=True)
        if not any(n.endswith(".safetensors") for n in manifest["qwen_files"]):
            raise ValueError("The source archive has no Qwen safetensors weights.")

        out.parent.mkdir(parents=True, exist_ok=True)
        out.mkdir()
        try:
            with TemporaryDirectory(prefix=".nisar-convert-", dir=out.parent) as tmp:
                legacy = Path(tmp) / "sar.pt"
                with old.open("sar.pt") as src, legacy.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024**2)
                payload = load_payload(legacy)
                detector, _ = checkpoint_spec(payload["detector"])
                verify_run(payload["linear"], payload["detector_sha256"],
                           detector["model"]["encoder.cls_token"].shape[-1])
                save_sar(payload, out / "sar.safetensors")
                del detector, payload
            for name in members:
                if name == "sar.pt":
                    continue
                dest = out / name
                dest.parent.mkdir(exist_ok=True)
                with old.open(name) as src, dest.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024**2)
            (out / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        except BaseException:
            shutil.rmtree(out)
            raise


def attach_adapter(source, adapter, out):
    files = [adapter/name for name in ADAPTER_FILES]
    if not all(p.is_file() and p.stat().st_size for p in files):
        raise ValueError("Pass the saved adapter directory containing adapter_config.json and adapter_model.safetensors.")
    config = json.loads(files[0].read_text(encoding="utf-8"))
    if not isinstance(config,dict) or config.get("peft_type") != "LORA":
        raise ValueError("Expected a PEFT LoRA adapter.")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Choose a new output path: {out}")
    with ZipFile(source) as old:
        manifest,members = bundle_members(old)
        if manifest.get("qwen_adapter_files"):
            raise ValueError("Use the original .nisar as input, without an embedded decoder adapter.")
        manifest["qwen_adapter_files"] = list(ADAPTER_FILES)
        out.parent.mkdir(parents=True,exist_ok=True)
        bytes_needed = sum(old.getinfo(n).file_size for n in members) + sum(p.stat().st_size for p in files)
        if shutil.disk_usage(out.parent).free < bytes_needed + 1024**2:
            raise OSError("Not enough free space for the new weights archive.")
        with TemporaryDirectory(prefix=".nisar-pack-",dir=out.parent) as work:
            staged = Path(work)/"weights.nisar"
            with ZipFile(staged,"w",compression=ZIP_STORED,allowZip64=True) as new:
                new.writestr("manifest.json",json.dumps(manifest,indent=2))
                for n in members:
                    print(f"Copying {n}",flush=True)
                    with old.open(n) as src,new.open(n,"w",force_zip64=True) as dst:
                        shutil.copyfileobj(src,dst,8*1024**2)
                for p in files:
                    new.write(p,f"qwen_adapter/{p.name}")
            with ZipFile(staged) as new:
                bundle_members(new)
                for n in members:
                    a,b=old.getinfo(n),new.getinfo(n)
                    if (a.CRC,a.file_size)!=(b.CRC,b.file_size):
                        raise ValueError(f"Copied weights differ: {n}")
            # NTFS mounts that cannot create hard links can still publish without overwrite.
            try:
                os.link(staged,out)
            except FileExistsError:
                raise
            except OSError:
                if shutil.disk_usage(out.parent).free < staged.stat().st_size:
                    raise OSError("This filesystem requires a second archive copy; choose an output on a drive with more free space.")
                with out.open("xb") as dst:
                    try:
                        with staged.open("rb") as src:
                            shutil.copyfileobj(src,dst,8*1024**2)
                    except BaseException:
                        dst.close()
                        out.unlink(missing_ok=True)
                        raise
    print(f"Saved {out} ({out.stat().st_size/2**30:.2f} GiB)")


def pack_single_weights(source, out, work_dir=None):
    """Export a model directory or a legacy archive as one safetensors file."""
    if out.suffix != ".safetensors":
        raise ValueError("Choose an output ending in .safetensors.")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Choose a new output path: {out}")
    if source.is_dir():
        pack_directory(source, out)
        return
    with TemporaryDirectory(prefix="nisar-single-", dir=work_dir or out.parent) as tmp:
        model = Path(tmp) / "model"
        convert_weights(source, model)
        pack_directory(model, out)


def unpack_bundle(path,root):
    """Extract the validated members into a fresh working directory."""
    root=Path(root)
    with ZipFile(path) as z:
        manifest,members=bundle_members(z)
        if shutil.disk_usage(root).free < sum(z.getinfo(n).file_size for n in members):
            raise OSError("Not enough free space to unpack the weights; set --work-dir.")
        for name in members:
            dest=root/name
            dest.parent.mkdir(exist_ok=True)
            with z.open(name) as src,dest.open("xb") as dst:
                shutil.copyfileobj(src,dst,8*1024**2)
    adapter=root/"qwen_adapter" if manifest.get("qwen_adapter_files") else None
    return root/"sar.pt",root/"qwen",adapter



def open_weights(source, work_dir):
    """Resolve a directory, legacy archive, or single safetensors export."""
    path = source
    if not path.is_dir():
        if path.suffix != ".safetensors":
            return unpack_bundle(path, work_dir)
        path = Path(work_dir) / "weights"
        path.mkdir()
        restore_weights(source, path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name in manifest_members(manifest, directory=True):
        if not (path / name).is_file():
            raise ValueError(f"Missing model file: {name}")
    adapter = path / "qwen_adapter" if manifest.get("qwen_adapter_files") else None
    return path / "sar.safetensors", path / "qwen", adapter


class SarModel:
    def __init__(self,weights,device_name):
        from change_model import (ChangeModel, TILE_SIZE, checkpoint_spec, inference_device,
                                  load_payload, scene_dtype, verify_run)
        pack=load_payload(weights)
        p,clip=checkpoint_spec(pack["detector"])
        self.spec=p["stats"]
        self.device,self.amp=inference_device(device_name,"auto")
        self.dtype=scene_dtype(self.device)
        mean,std=(np.asarray(self.spec[k],dtype=np.float32).reshape(2,1,1) for k in ("mean","std"))
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
            raise ValueError("Invalid SAR normalization.")
        self.normalization=mean,std,clip
        self.linear=pack["linear"]
        self.model=ChangeModel(TILE_SIZE,p["arch"],p["head_width"],grad_checkpoint=False)
        verify_run(self.linear,pack["detector_sha256"],self.model.encoder.dim)
        self.model.load_state_dict(p["model"],strict=True)
        self.model.to(self.device).eval()
        # Keep the encoder dtype stable across detection and classification requests.
        self.model.encoder.to(dtype=self.dtype)

    def detect(self,size,pair,*,batch_size,overlap,threshold):
        from change_model import infer_scene
        return infer_scene(self.model,size,pair,*self.normalization,self.device,self.amp,
                           batch_size,overlap,threshold)

    def classify(self,size,pair,mask):
        from change_model import build_scene_prefix, classify
        prefix,report=build_scene_prefix(self.model.encoder,size,pair,mask,self.normalization,
                                        self.device,self.dtype,64,.98)
        return classify(prefix,report,self.linear)


def output_parts(out):
    if isinstance(out,tuple):
        if len(out)!=2:
            raise ValueError("Unsupported Qwen vision tuple.")
        return out[0],out[1]
    return out.pooler_output,out.deepstack_features


def join_vision(first,last,cat):
    a,sa=output_parts(first)
    b,sb=output_parts(last)
    if len(sa)!=len(sb):
        raise ValueError("Qwen DeepStack stream count changed.")
    main=cat((a,b),dim=0)
    deep=[cat((x,y),dim=0) for x,y in zip(sa,sb,strict=True)]
    if isinstance(first,tuple) and isinstance(last,tuple):
        return main,deep
    extra={}
    if getattr(first,"last_hidden_state",None) is not None:
        extra["last_hidden_state"]=cat((first.last_hidden_state,last.last_hidden_state),dim=0)
    return replace(first,pooler_output=main,deepstack_features=deep,**extra)


class QwenModel:
    def __init__(self,root,device_name,adapter=None):
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        from change_model import inference_device, scene_dtype
        self.torch=torch
        self.device,_=inference_device(device_name,"auto")
        self.dtype=scene_dtype(self.device)
        self.processor=AutoProcessor.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        self.lm=Qwen3VLForConditionalGeneration.from_pretrained(
            root,local_files_only=True,use_safetensors=True,dtype=self.dtype,
            attn_implementation="sdpa").to(self.device).eval()
        self.adapter_model=None
        self.has_adapter=adapter is not None
        if self.has_adapter:
            from peft import PeftModel
            self.adapter_model=PeftModel.from_pretrained(self.lm,adapter,is_trainable=False).eval()
            self.lm=self.adapter_model.get_base_model()
            logging.info("Loaded decoder LoRA from %s",adapter)
        config=self.lm.config.vision_config
        if (config.patch_size,config.spatial_merge_size)!=(QWEN_PATCH,QWEN_MERGE):
            raise ValueError("Expected Qwen3-VL's 16-pixel patches and 2×2 spatial merge.")
        self.vision=self.lm.model.visual

    def encode(self,pictures):
        with self.torch.inference_mode():
            batch=self.processor.image_processor(images=pictures,do_resize=False,return_tensors="pt").to(self.device)
            grids,counts=panel_grid(pictures)
            if batch["image_grid_thw"].tolist()!=grids:
                raise ValueError("Qwen changed the cached image grid.")
            out=self.vision(batch["pixel_values"].to(dtype=self.dtype),grid_thw=batch["image_grid_thw"])
            main,deep=output_parts(out)
            if main.shape[0]!=sum(counts) or not deep:
                raise ValueError("Cached vision features are missing image tokens or DeepStack streams.")
            if self.device.type=="cuda":
                self.torch.cuda.synchronize(self.device)
        return out

    def report(self,pictures,result,trace,cached_pair=None):
        torch=self.torch
        if len(pictures) != 3 or len({p.size for p in pictures}) != 1:
            raise ValueError("Expected aligned T0, T1, and standalone mask panels.")
        grids,counts=panel_grid(pictures)
        trace.record_panels(pictures)
        cache=None
        if cached_pair is not None:
            try:
                with torch.inference_mode():
                    cache=join_vision(cached_pair,self.encode(pictures[2:]),torch.cat)
            except (RuntimeError,AttributeError,TypeError,ValueError):
                logging.exception("Could not combine cached vision features; encoding all panels together.")
        messages=evidence_messages(result)
        prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=self.processor(text=[prompt],images=pictures,do_resize=False,return_tensors="pt").to(self.device)
        if batch["image_grid_thw"].tolist()!=grids or int((batch["input_ids"]==self.lm.config.image_token_id).sum())!=sum(counts):
            raise ValueError("Qwen processor grid and image placeholders do not match.")
        batch={k:v.to(dtype=self.dtype) if v.is_floating_point() else v for k,v in batch.items()}
        n=batch["input_ids"].shape[1]
        trace.attempts.append(dict(prompt=prompt,input_token_count=n,image_grid_thw=grids,
                                   visual_tokens_observed=False,reused_t0_t1_features=cache is not None))
        def observed(_module,_args,out):
            main,deep=output_parts(out)
            if main.shape[0]!=sum(counts):
                raise ValueError("Qwen output token count does not match the prompt.")
            trace.record_streams(main,list(deep))
        def reuse(*args,grid_thw=None,**kwargs):
            if grid_thw is None and len(args)>1:
                grid_thw=args[1]
            if grid_thw is None or grid_thw.tolist()!=grids:
                raise ValueError("Unexpected vision grid for cached Qwen inputs.")
            return cache
        with torch.inference_mode():
            handle=self.vision.register_forward_hook(observed)
            # The checkpoint can carry sampling defaults even for greedy reports.
            greedy=dict(max_new_tokens=256,do_sample=False,num_beams=1,
                        repetition_penalty=1.08,
                        temperature=1.0,top_p=1.0,top_k=50)
            if self.has_adapter:
                greedy["eos_token_id"]=self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            try:
                if cache is None:
                    output=self.lm.generate(**batch,**greedy)
                else:
                    with patch.object(self.vision,"forward",new=reuse):
                        output=self.lm.generate(**batch,**greedy)
            finally:
                handle.remove()
        raw=self.processor.batch_decode(output[:,n:],skip_special_tokens=True)[0].strip()
        trace.attempts[-1]["raw_answer"]=raw
        return raw

