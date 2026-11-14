# NISAR SAR Change Explorer

The browser UI has separate Pick and Results files (`pick.css`/`pick.js` and
`results.css`/`results.js`). `inference.py` loads the SAR and Qwen models once,
then serves the UI and analysis API with FastAPI. Run one server process: the
loaded models and current pair are shared in memory.

## Run on the model host

Use Python 3.11 or newer. Install a PyTorch build that matches the host's GPU
from the [official PyTorch selector](https://pytorch.org/get-started/locally/),
then install the remaining dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build selected for this machine first.
python -m pip install -r requirements.txt
```

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