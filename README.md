# NISAR SAR Change Explorer

The browser UI has separate Pick and Results files (`pick.css`/`pick.js` and
`results.css`/`results.js`). `inference.py` loads the SAR and Qwen models once,
then serves the UI and analysis API with FastAPI. Run one server process: the
loaded models and current pair are shared in memory.

## Who this is for

Develop the website for **environmental analysts** and **enthusiasts**. Analysts
need precise selections, reproducible settings, measured evidence, and exports.
Enthusiasts need approachable before/after exploration and clear explanations
without assumed radar knowledge. Both use the same workflow: explain the task
first, then reveal technical detail when wanted. These are product personas,
not separate sign-in roles. [AGENTS.md](AGENTS.md) records this design context
for future development.

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
first tab has two steps: load the GCOV files, then select an area.
Before/After controls preview either date on the same map grid, preserving
zoom, pan, and the selected area. Colour and grayscale work with either date.
The Run button stays at the top of the compact selection sidebar. **How analysis
works & map details** opens a dialog with the workflow explanation, selection
measurements, image resolution, analysis cell size, and CRS. The Frequency label
opens an explanation of the GCOV radar-data group used by the model, with a
link to the [NISAR GCOV guide](https://asfopensarlab.github.io/NISAR_Cookbook/gcov-intro/).
The vertical threshold slider applies to the next model run, starting at the server's
`--threshold` value (default 0.1). Lower values include more candidate changes;
higher values keep only higher-scoring detections. These scores are not
calibrated probabilities or a direct measure of physical significance.
The small numeric field accepts values from 0.01 to 0.99 in steps of 0.01 and
stays synchronized with the slider. Invalid or empty values disable the run.

Runs use a dedicated progress view, with no setup sidebar beneath or beside it.
They show the current processing stage, elapsed time, and actual processed-tile
progress during detection. Other stages use an indeterminate progress bar.
Reloading the same browser tab reconnects to its running analysis. The Change
viewer has independent **Show change** and **HH grayscale** switches; the blue
detector overlay stays blue in grayscale mode and works in T0, T1, and split
views. Results links open this same viewer.
The Results map also has a Before/After date selector and independent
**Show change** and **HH grayscale** switches. Both switches keep their state
when changing dates. The mask stays blue in grayscale, and PNG downloads use
the selected date, grayscale setting, and mask state. The zoom controls are
**+** and **−**.

The UI starts an analysis with `POST /infer` (JSON: `box`, `revision`, optional
`threshold`). It returns HTTP 202 with a `run_id`. Poll `GET /infer/{run_id}`
for `status`, `stage`, `message`, `elapsed_s`, and optional `completed`/`total`
tile counts. A completed job includes `result`; a failed job includes `error`.
The server accepts one active analysis at a time and retains eight recent job
statuses in memory. `GET /result` still returns the latest completed report.

Progress includes an approximate remaining-time range once there is enough
measured work. Tile throughput estimates the detection stage; image-write
throughput estimates final rendering. With similar completed runs from the
same server, the estimate can include the remaining stages. Estimates use up
to eight recent in-memory timing profiles, adjust image-based stages for ROI
size, and stay unavailable when there is no useful basis. They are rough
ranges, not confidence intervals or promised completion times.

Results and Change viewer share grayscale, mask visibility, date, divider,
and normalized map focus/zoom through browser session storage, scoped to the
result revision. EPSG labels open an explanation of the map coordinate system,
including its UTM/polar name and why it matters for projected area and GIS use.

The evidence section compares flagged and other valid cells by HH/HV channel,
showing signed dB changes, group sizes, and differences between group medians.
Scientific details expose usable model coverage, valid and unflagged grid
areas, native-date medians, all nine sector flag rates, acquisition metadata,
projected bounds, preprocessing, detector settings, and checkpoint identifier
when available. Missing fields remain explicitly unrecorded. Whole-selection
differences between date medians are distinguished from medians of paired
cell changes. Areas use the projected grid; no validated accuracy or causal
attribution is implied. See the [NISAR projection guide](https://nisar-docs.asf.alaska.edu/projections/)
and [GCOV data guide](https://nisar-docs.asf.alaska.edu/gcov/) for product context.

Download the full report and measurements as JSON or the leading three sector
comparisons as CSV. The CSV includes both channels, sample counts, signed dB
statistics, acquisition times, threshold, spacing, and EPSG code.

Run the regression checks without loading the weight bundle:

```bash
python -m unittest -v test_analysis
node test_explorer.js
```
