# NISAR SAR Change Explorer

The browser UI has separate Pick and Results files (`pick.css`/`pick.js` and
`results.css`/`results.js`). `inference.py` loads the SAR and Qwen models once,
then serves the UI and analysis API with FastAPI. Run one server process: the
loaded models and current pair are shared in memory.

## Run on the model host

Use Python 3.11 or newer. Install a PyTorch build for the host from the
[official PyTorch selector](https://pytorch.org/get-started/locally/), then
install the remaining dependencies. For a CPU-only machine, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

On a GPU host, replace the CPU-only PyTorch command with the command selected
for that GPU. The model bundle in this checkout is roughly 6.4 GiB, and CPU
loading uses float32, so allow substantially more free RAM than the bundle's
file size. CPU inference can take a long time; a 15 GiB RAM machine may run out
of memory while loading or analyzing. A CPU-only browser can instead connect
to a teammate's GPU-hosted server.

Keep the weight bundle and GCOV files on the host. To start with a prepared
pair, run:

```bash
python inference.py --weights /path/to/model.safetensors \
  --t0 /path/to/earlier.h5 --t1 /path/to/later.h5 \
  --port 8000
```

The weight path can also be a supported `.nisar` archive or model directory.
Omit `--t0` and `--t1` if users should upload a pair through the browser. Use
`--work-dir /path/to/large/disk` when uploads or model extraction need more
space. The server prints its URL after both models load.

For a CPU-only local check, use a small selection after the server starts:

```bash
python inference.py --weights ./model.safetensors \
  --sar-device cpu --qwen-device cpu --batch-size 1 \
  --decoder-overview-size 256 --port 8000
```

Open `http://127.0.0.1:8000`, upload an earlier and later NISAR L-SAR GCOV
`.h5` pair, select a small area, and click **Run model on selection**. The
weights alone do not contain the GCOV input files. Press Ctrl+C to stop the
server. To avoid uploading, also pass `--t0 /path/to/earlier.h5` and
`--t1 /path/to/later.h5` to the same command.

The three tabs are **Pick an area**, **Change viewer**, and **Results**. The
first tab has two steps: load the GCOV files, then select an area. Its vertical
threshold slider applies to the next model run, starting at the server's
`--threshold` value (default 0.1). Lower values include more candidate changes;
higher values keep only higher-scoring detections. These scores are not
calibrated probabilities or a direct measure of physical significance.

Runs show the current processing stage, elapsed time, and actual processed-tile
progress during detection. Other stages use an indeterminate progress bar.
Reloading the same browser tab reconnects to its running analysis. The Change
viewer has independent **Show change** and **HH grayscale** switches; the blue
detector overlay stays blue in grayscale mode and works in T0, T1, and split
views. Results links open this same viewer.

The UI starts an analysis with `POST /infer` (JSON: `box`, `revision`, optional
`threshold`). It returns HTTP 202 with a `run_id`. Poll `GET /infer/{run_id}`
for `status`, `stage`, `message`, `elapsed_s`, and optional `completed`/`total`
tile counts. A completed job includes `result`; a failed job includes `error`.
The server accepts one active analysis at a time and retains eight recent job
statuses in memory. `GET /result` still returns the latest completed report.

Run the regression checks without loading the weight bundle:

```bash
python -m unittest -v test_analysis
```
