# NISAR SAR Change Explorer

The code is organized by feature. Python features live in `nisar/features/`,
and browser features live in `frontend/features/`. The application loads the
SAR and Qwen models once, then serves the UI and analysis API with FastAPI.
Run one server process: the loaded models and current pair are shared in memory.

## Start here: live models or saved sample

```bash
python launch.py
```

The launcher asks whether this machine can run the models. Answer **no** for a
machine with less than 16 GB RAM and no GPU. It starts with the saved sample,
without importing PyTorch, Transformers, NumPy, or HDF5. Answer **yes** to start
the existing live inference workflow. The launcher defaults to sample mode
when run without an interactive terminal.

For judges, only the lightweight server dependencies are needed:

```bash
python -m pip install -r requirements-sample.txt
python launch.py --mode sample
```

Open `http://127.0.0.1:8000`. **Explore the case study** is the default entry
when a saved sample is available, including in live model mode. It introduces
the Karakalpakstan study landscape, acquisition dates, and a geographic locator.
**Data and methods** expands the fixed filenames, ROI, threshold, and provenance.
Click **Explore the case study** to use the same Results and
Change viewer, including date switching, grayscale, overlays, zoom, and report
downloads. The sample is labeled as a saved run. The Results page also offers
a three-stop guided investigation: stronger radar return in the
middle-left sector, a mostly unflagged center sector, and an unresolved alert
in the upper-right sector. Choose a stop to focus its gold sector outline and
read the saved measurements, limitations, and suggested independent checks.
The guide preserves date, grayscale, and detector-overlay settings. Use
**Full-area overview** to return to exploration. These are sector comparisons,
not independently validated reference sites; the tour is limited to this pair
and selection.

**Independent regional context** presents SIC ICWC's published Sudochye
open-water figures and dated Landsat reference maps (27 June and 23 September
2026), alongside their gaps to the radar dates. Extracted reference maps and
table values are bundled for offline playback; the full source reports are linked.
Maps have different
extents and are not aligned to the SAR grid; their blue water classification
is the publisher's, not the detector overlay. A regional source does not
validate individual radar flags or the flagged area. Location-level optical
verification remains incomplete. Source IDs, dates, geographic scope, and
checksums accompany the JSON, Markdown, and plain-text reports. Context is
attached separately at playback; the immutable inference bundle is unchanged.

The sample's inputs and selection
cannot be edited; live upload, preparation, and inference endpoints are
disabled in sample mode. Sample playback requires neither the weight bundle
nor the source `.h5` files.

The requested sample uses these two files from `src/`:

| Date | GCOV filename |
| --- | --- |
| Before · T0 | `NISAR_L2_PR_GCOV_023_107_D_067_2005_DHDH_A_20260621T150729_20260621T150804_P05023_N_F_J_001.h5` |
| After · T1 | `NISAR_L2_PR_GCOV_030_107_D_067_2005_DHDH_A_20260913T150726_20260913T150801_P05023_N_F_J_001.h5` |

The ROI is **left 5323, top 2414, right 11255, bottom 6066** in native T0
pixels, matching the supplied screenshot. Right and bottom are exclusive:
**5,932 × 3,652 px**, with **threshold 0.10**. The cache records exact source
filenames, acquisition metadata, SHA-256 checksums, generation time, model
settings, measurements, and real model output. JSON and text report exports
include the sample provenance.

On a machine with the models and source files, generate the sample once:

```bash
python cache_sample.py
```

This runs real inference on the fixed pair, ROI, and threshold and writes an
immutable, checksummed bundle under `samples/judges/`. Later invocations reuse
the bundle. Use `--refresh` to generate a replacement; the previous bundle
remains available on disk. The `current.json` pointer changes only after all
artifacts are saved. Model/device options can be passed through, for example:

```bash
python cache_sample.py --refresh --sar-device cpu --qwen-device cpu --batch-size 2
```

An existing one-shot export from this exact pair/ROI can be imported with
`python cache_sample.py --from-export /path/to/export`. The builder checks ROI,
threshold, acquisition metadata, and image dimensions before publishing.
Source checksums identify the supplied inputs; import provenance relies on the
export having been produced by the documented inference command.
If Qwen produces usable text but misses the structured report format, the sample
shows its original sentences as a clearly labeled, unverified plain-text AI
explanation. It does not invent missing uncertainty fields or validated claims.
The live workflow keeps its strict structured validator. Empty, broken JSON,
or unusable responses still leave the measurements available.

To update an existing sample's presentation without models or source files:

```bash
python cache_sample.py --refresh-report
```

This publishes a new checksummed revision, preserving the original model
response, measurements, all image bytes, and the previous bundle. Restart a
running sample server to load the new revision.

Live model mode also offers the optional saved sample without replacing the
currently loaded pair or live result:

```bash
python launch.py --mode model --t0 /path/to/earlier.h5 --t1 /path/to/later.h5
```

`python inference.py` and `python -m nisar` continue to start live model mode
directly. Use `--cache-dir /path/to/cache` with the launcher/builder, or
`--sample-dir /path/to/cache` with `inference.py`, to change the sample location.

## Project structure

```text
nisar/
  app/                    # CLI, HTTP API, browser routes, resource paths
  features/
    imagery/              # GCOV grids, geographic alignment, preview rendering
    analysis/             # Pair/selection workflow and progress estimates
    reporting/            # Measurements, evidence, AI claim contract, exports
    samples/              # Fixed sample specification, cache builder, playback
  infrastructure/models/  # SAR/Qwen runtimes and weight conversion
frontend/
  app/                    # Main HTML shell joining selection and results
  features/
    selection/            # Upload, dates, area selection, run setup/progress
    change_viewer/        # Before/after comparison and full-image popup
    results/              # Result map, scientific details, CSV/JSON exports
    reporting/            # Evidence and interpretation presentation
    samples/              # Read-only Step 0 sample details and entry point
  shared/                 # Map viewport/state, navigation styles, icons
tests/
  analysis/               # Detector, selection, progress, and job API checks
  reporting/              # Evidence contract and report export checks
  app/                    # Browser resource integration checks
  samples/                # Cache integrity, read-only API, lightweight startup
  frontend/               # JavaScript state, measurement, and export checks
  support.py              # Shared synthetic data, model stubs, ASGI harness
docs/                     # Architecture guide and report contract
models/                   # Local model weights (ignored by Git)
samples/judges/           # Portable saved sample and provenance (included)
src/                      # Source GCOV pair (ignored by Git)
launch.py                 # Interactive model/sample startup chooser
cache_sample.py           # Generate the fixed sample once, then reuse it
inference.py              # Backward-compatible command entry point
```

See [the architecture guide](docs/ARCHITECTURE.md) for feature ownership,
dependencies, and where to add code. Browser URLs remain stable; the explicit
resource map in `nisar/app/web.py` resolves them to their feature folders.

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

Run commands from the repository root. Keep the weight bundle in `models/`
and GCOV files on the host. To start with a prepared pair, run:

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
python inference.py --weights ./models/model.safetensors \
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

**Evidence & interpretation** presents a short main finding, sourced numeric
evidence, unverified AI observations, one optional hypothesis with its open
assumptions, and independent next checks. Coverage/flag composition and signed
HH/HV comparison charts use application-calculated values. The sector chart
can focus that sector in the Results map. Reference chips reveal the source
measurements without repeating the full scientific-details narrative.

The report can be downloaded as **Markdown (.md)** or **plain text (.txt)**.
`GET /report.md?v=REV` and `/report.txt?v=REV` check that the requested result
is still current. Exports preserve claim labels, evidence IDs, exact numbers,
dates, method scope, and provenance. Charts are represented by their values
in the portable text export.

Qwen is prompted for bounded JSON claims with evidence references; numeric
measurements remain application-generated. Invalid or legacy prose output
keeps the measured report available while marking structured AI interpretation
unavailable. Raw model output remains in the technical inspector. This checks
format and references, not scientific correctness. The adapted Qwen model
still needs a real-pair evaluation of the new output contract.

[REPORT_CONTRACT.md](docs/REPORT_CONTRACT.md) describes the schema and the integration
boundary for future chat and web research. Live retrieval and chat are not enabled.
The saved case study includes reviewed regional source snapshots; suggested
location-level verification checks remain labeled as not completed.

Run the regression checks without loading the weight bundle:

```bash
python -m unittest discover -s tests -t . -v
node tests/frontend/test_explorer.js
```
