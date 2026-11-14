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

## Give teammates access

For teammates in different locations, connect the host and both teammates to
the same Tailscale network. On the host, leave `inference.py` bound to its
default `127.0.0.1` address and, in another terminal, run:

```bash
tailscale serve 8000
```

Share the `https://...ts.net/` URL printed by Tailscale. Both teammates open
that URL in their browsers; the model runs only on the host. Tailscale Serve
limits the URL to authorized members of that network. See the
[Tailscale Serve guide](https://tailscale.com/docs/use-cases/application-testing/share-local-dev-server-with-team).

On a trusted local network, `python inference.py ... --host 0.0.0.0 --port 8000`
also works; teammates then open `http://HOST_LAN_IP:8000`. This demo has no
application login, so do not expose that port directly to the public internet.

**Shared-state limit:** the current pair and latest result belong to the whole
server. If one person uploads a new pair or starts a new inference, an earlier
result can become unavailable for everyone. Coordinate runs or use separate
server instances when independent simultaneous work is needed. Keep one
Uvicorn worker per instance, because each worker would load its own models and
have different in-memory state.
