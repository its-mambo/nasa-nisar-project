"""Lazy Qwen3-VL loading, with one GPU KV cache per active conversation."""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import monotonic
import shutil
import os


SYSTEM = (
    "Answer questions about this selected radar result. The three panels are clean T0, clean T1, "
    "and a standalone detector mask, in that order. Clean images show HH in red and HV in green. "
    "On the mask white means flagged, black is valid "
    "and unflagged, and gray is no paired data. At display resolution white means at least one "
    "flagged cell. If metadata says saved_overlay_footprint, the third panel only shows visible saved flags: "
    "black may also include no data or an invisible flag. Use measured counts for totals. "
    "The panel gives location, not sign or cause. In measured metadata, HH/HV medians are paired "
    "T1 minus T0 in dB; negative is weaker, positive stronger. Sector medians are not individual "
    "patch measurements. Separate observations from hypotheses. Do not assert a cause, flood extent, "
    "or land-cover class without independent evidence. State what cannot be inferred and answer "
    "concisely. For questions unrelated to this result, say what you can help with."
)
ASSETS = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.json", "chat_template.jinja", "vocab.json", "merges.txt",
)
REQUIRED = set(ASSETS[:6])
CONTEXT_LIMIT = 4096
REPLY_LIMIT = 384
SESSION_LIMIT = 2
SESSION_TTL = 900


def add_chat_options(parser):
    from nisar.app.paths import MODEL_ROOT

    parser.add_argument("--qwen-weights", type=Path, default=MODEL_ROOT / "qwen3-vl-2b.safetensors",
                        help="Local Qwen3-VL-2B-Instruct weights for Results chat.")
    parser.add_argument("--qwen-assets", type=Path, default=MODEL_ROOT / "qwen3-vl-2b",
                        help="Local Qwen config, tokenizer and processor directory.")
    parser.add_argument("--qwen-device", help="Device for chat; defaults to the SAR device or auto in saved mode.")
    parser.add_argument("--no-chat", action="store_true", help="Disable Qwen chat.")


def configure_chat(args, default_device="auto"):
    if args.no_chat:
        return None, "Qwen chat was disabled with --no-chat. Restart without that flag to enable it."
    if not args.qwen_weights.is_file():
        return None, ("Qwen weights were not found at " + str(args.qwen_weights) +
                      ". Put the weights there or set --qwen-weights, then restart the server.")
    return QwenChat(args.qwen_weights, args.qwen_assets, args.qwen_device or default_device), None


class ChatUnavailable(ValueError):
    pass


@dataclass
class Session:
    cache: object
    rope: object
    touched: float


class QwenChat:
    def __init__(self, weights, assets, device):
        self.weights, self.assets, self.device = Path(weights), Path(assets), device
        self.lock = Lock()
        self.sessions = OrderedDict()
        self.model = None

    def load(self):
        if self.model is not None:
            return
        missing = sorted(name for name in REQUIRED if not (self.assets / name).is_file())
        if missing:
            raise ChatUnavailable(f"Qwen assets are missing in {self.assets.name}: {', '.join(missing)}. See README.")
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except (ImportError, RuntimeError) as exc:
            raise ChatUnavailable("Install requirements-chat.txt and a torchvision build matching PyTorch.") from exc
        device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                              if self.device == "auto" else self.device)
        if device.type == "cuda":
            with torch.cuda.device(device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        with TemporaryDirectory(prefix="nisar-qwen-") as staging:
            root = Path(staging)
            for name in ASSETS:
                if (self.assets / name).is_file():
                    shutil.copyfile(self.assets / name, root / name)
            # Transformers expects model.safetensors next to config.json. Pass the
            # actual file directly to the loader via a link, with no multi-GB copy.
            source = self.weights.resolve(strict=True)
            target = root / "model.safetensors"
            try:
                target.symlink_to(source)
            except OSError:
                try:
                    os.link(source, target)           
                except OSError:
                    shutil.copyfile(source, target)    
            processor = AutoProcessor.from_pretrained(root, local_files_only=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                root, dtype=dtype, local_files_only=True, use_safetensors=True,
            ).to(device).eval()
        self.device = device
        self.torch, self.processor, self.model = torch, processor, model

    def step(self, ids, cache=None, rope=None, **inputs):
        start = cache.get_seq_length() if cache is not None else 0
        if rope is not None:
            self.model.model.rope_deltas = rope
        with self.torch.inference_mode():
            return self.model(
                input_ids=ids, cache_position=self.torch.arange(start, start + ids.shape[-1], device=ids.device),
                past_key_values=cache, use_cache=True, logits_to_keep=1, **inputs,
            )

    def turn(self, revision, client, question, context):
        """Yield events. Commit the updated cache only after a complete reply."""
        if not self.lock.acquire(blocking=False):
            yield {"error": "Qwen is already answering. Try again in a moment."}
            return
        try:
            yield from self._turn(revision, client, question, context)
        finally:
            self.lock.release()

    def _turn(self, revision, client, question, context):
        now = monotonic()
        for key, entry in list(self.sessions.items()):
            if key[0] != revision or now - entry.touched > SESSION_TTL:
                del self.sessions[key]
        key = (revision, client)
        # The cache mutates in place on forward. A cancelled or failed turn must
        # discard it, otherwise the next question would see a partial response.
        previous = self.sessions.pop(key, None)
        self.load()
        from transformers import TextStreamer
        chunks = []

        class Streamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                if text:
                    chunks.append(text)

        tokenizer = self.processor.tokenizer
        for token in tokenizer.all_special_tokens:
            question = question.replace(token, " ")
        streamer = Streamer(tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        encode = lambda text: tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        suffix = encode("\n<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n")
        if previous and previous.cache.get_seq_length() + suffix.shape[-1] + REPLY_LIMIT + 1 <= CONTEXT_LIMIT:
            output = self.step(suffix, previous.cache, previous.rope)
            yield {"status": "reused"}
        else:
            del previous
            images, metadata = context()
            content = []
            for label, image in zip(("T0: earlier acquisition", "T1: later acquisition", "Detector mask"), images):
                content += [{"type": "text", "text": label + "\n"}, {"type": "image", "image": image}]
            content.append({"type": "text", "text": "\nMeasured metadata: " + metadata + "\nQuestion: " + question})
            yield {"status": "prefill"}
            batch = self.processor.apply_chat_template(
                [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                 {"role": "user", "content": content}],
                tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            ).to(self.device)
            ids = batch.pop("input_ids")
            if ids.shape[-1] + REPLY_LIMIT + 1 > CONTEXT_LIMIT:
                raise ChatUnavailable("Image context exceeds the configured chat token budget.")
            output = self.step(ids, **batch)
            del batch, images, content
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos = self.model.generation_config.eos_token_id
        stop = set(eos if isinstance(eos, list) else [eos]) | {im_end}
        for _ in range(REPLY_LIMIT):
            token = output.logits[:, -1, :].argmax(dim=-1)
            if token.item() in stop:
                break
            streamer.put(token.cpu())
            while chunks:
                yield {"delta": chunks.pop(0)}
            output = self.step(token[:, None], output.past_key_values, output.rope_deltas)
        streamer.end()
        while chunks:
            yield {"delta": chunks.pop(0)}
        # Close the assistant turn on EOS or at the reply limit so the next
        # incremental ChatML prompt starts at the exact continuation position.
        output = self.step(self.torch.tensor([[im_end]], device=self.device),
                           output.past_key_values, output.rope_deltas)
        self.sessions[key] = Session(output.past_key_values, output.rope_deltas, monotonic())
        while len(self.sessions) > SESSION_LIMIT:
            self.sessions.popitem(last=False)
        yield {"done": True}
