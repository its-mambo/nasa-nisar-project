# NISAR SAR Change Explorer

Compare two NISAR L2 GCOV acquisitions over the same area. The app aligns the paired HH/HV imagery, runs a learned change detector on overlapping tiles, and reports where it flagged a change in radar backscatter.

Read the [project overview](docs/PROJECT_OVERVIEW.md) and
[detector evidence and limitations](docs/MODEL_CARD.md). The prototype supports
investigation; its detector flags have not been independently validated as
environmental change.

The saved case now includes [four location-matched optical checks](evaluation/README.md)
with nearby Landsat dates, original source crops, local radar measurements and
explicit interpretation failures. This exploratory pilot does not establish
detector accuracy.

## Explore the saved sample (start here)
For correct work of the application, first donwload two models' weights:
**detector.safetensorshttps**(custom pretrained seg-model): https://huggingface.co/EscRq1/nisar-seg-vith
**models.safetensors**(Qwen-VL 2B): https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

Put both **.safetensors** files directly into **/models** folder.
**Optional:** if you want to test manual change detection, either download a pair of .h5 files from ASF Vertex on different dates, or download prepared pair:
https://huggingface.co/datasets/EscRq1/netherlands-nisar-pair

Use **Python 3.11 or newer**. Clone or download the complete repository, including
`samples/judges`, then run these commands from its root:

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements-sample.txt
python launch.py --mode sample --no-chat
```

Open [the local app](http://127.0.0.1:8000) and select **Explore the case study →**.
The Before/After images, detector overlay, measurements, and reports are bundled
in Git. No GPU, checkpoint, source HDF5 files, or cache generation is needed.
After installation, playback works offline. The lightweight install supplies
only the web server; optional model and chat dependencies are separate.

To verify a fresh install, run:

```bash
python -m unittest tests.samples.test_playback -v
```

This starts a temporary server, checks every saved PNG against its manifest,
and checks the pages and report export using only the lightweight dependencies.
See [sample details and troubleshooting](samples/README.md) if the dates or images
are missing or the Explore button is disabled.

## Run with your detector

Install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

If you have the original packed `model.safetensors`, extract the detector once:

```bash
python extract_detector.py --weights /path/to/model.safetensors --out models/detector.safetensors
python launch.py --mode model --sar-device cuda:0
```

Open [the local app](http://127.0.0.1:8000). To select a different checkpoint, pass `--weights /path/to/detector.safetensors`. The extractor only requires Python, copies the original detector tensors and settings without conversion, and leaves the source file untouched. The two image branches share one ViT-H encoder; the output includes that encoder and its CNN change head.

Upload a Before and After GCOV HDF5 pair, select an area in the common footprint, then run analysis. For a direct local run with native T0 pixel coordinates:

```bash
python inference.py --sar-device cuda:0 --t0 before.h5 --t1 after.h5 \
  --roi 1000 1000 2500 2500 --threshold 0.1 --output-dir run-output
```

Use `python inference.py --help` for device, tile batching, thresholds, and upload settings. Live mode is local by default. The detector file and source products are not included in the repository.

## Read the results

Switch between Before and After, toggle the change overlay, or inspect HH/HV sector statistics. Blue marks detector flags; opacity follows the magnitude of the HH difference. The flag area counts grid cells, and unflagged pixels may still have changed. The report includes measurements, acquisition warnings and suggested independent checks. Detector flags alone do not establish water, flooding, or a physical cause.

**Ask about this result** uses optional local Qwen chat for both live and saved results. See setup below.

See [architecture](docs/ARCHITECTURE.md) and the [report contract](docs/REPORT_CONTRACT.md).

## Optional Qwen chat

The detector runs independently. To enable image-aware chat in Results, place the original Qwen3-VL-2B-Instruct weights at `models/qwen3-vl-2b.safetensors`. The model also needs its original tokenizer, config, and processor files. Install `torch` and its matching `torchvision` build from the same PyTorch wheel index, then run:

```bash
python -m pip install -r requirements-chat.txt
hf download Qwen/Qwen3-VL-2B-Instruct --include '*.json' '*.txt' '*.jinja' \
  --local-dir models/qwen3-vl-2b
python launch.py
```

The download filter fetches supporting files only. Open a saved sample or choose **y** and run a live analysis. Qwen loads on the first question; inference is local and replies stream into the chat. The first question includes clean T0/T1 images, the detector mask (or the visible overlay footprint for older saved samples), and measured metadata. Follow-ups reuse the conversation's in-memory attention KV cache. The cache is bounded to two active conversations and 4,096 tokens per conversation. An idle, evicted, or full context starts fresh on the next question; the cache is not written to disk. A new analysis gets its own chat.

Chat defaults to the detector's device in live mode, or CUDA if available when opening only a saved sample. On a multi-GPU machine, use `python launch.py --qwen-device cuda:1` to put Qwen on a separate device. Custom paths use `--qwen-weights` and `--qwen-assets`. Use `--no-chat` to disable chat. When Qwen is unavailable, the chat displays the cause.
