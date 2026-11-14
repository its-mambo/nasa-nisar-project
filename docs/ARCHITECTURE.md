# Feature-based modular architecture

The application remains one Python process and one shared browser workflow.
Features own related behavior, with small composition and infrastructure layers.
Python uses the `nisar` package and absolute imports. The browser uses ordinary
scripts in their existing load order; this reorganization adds no build step.

## Backend ownership

| Location | Responsibility | Main modules |
| --- | --- | --- |
| `nisar/features/imagery/` | Read GCOV data, align dates, render preview pixels | `gcov.py`, `rendering.py` |
| `nisar/features/analysis/` | Manage the loaded pair and selection, orchestrate detection and reporting, estimate progress | `service.py`, `progress_timing.py` |
| `nisar/features/reporting/` | Derive measurements and patterns, prepare decoder evidence, validate AI claims, export reports | `roi_report.py`, `roi_interpretation.py`, `roi_visual.py`, `qwen_evidence_prompt.py`, `evidence_report.py` |
| `nisar/infrastructure/models/` | Implement model loading, inference adapters, and weight packaging | `change_model.py`, `dual_model.py` |
| `nisar/app/` | Parse commands, create models/services, adapt HTTP requests, serve browser resources | `cli.py`, `server.py`, `web.py`, `paths.py` |

`analysis.service.Demo` owns the current pair, revision, model references, and
latest result. It coordinates imagery and reporting, receiving loaded SAR/Qwen
objects from the CLI. Features do not import the application layer. The model
adapter uses the reporting prompt and visual contract to call Qwen; reporting
does not import model implementations. Imagery has no dependency on analysis.
Package initializers only describe their modules and do not load models.

`server.py` is the HTTP adapter: it handles request validation, the shared model
lock, and asynchronous job status. Domain calculations belong in feature
modules. `web.py` maps existing public URLs to files under `frontend/` using
an allowlist, preserving script order and avoiding a public mount of the
repository. Resource paths derive from `paths.py`, independent of the current
working directory.

## Browser ownership

`frontend/app/index.html` composes the selection and results views. Screen
scripts and styles live together under `frontend/features/`:

- `selection/`: file upload, date preview, area selection, settings, run progress.
- `change_viewer/`: before/after/split comparison and the full-image popup.
- `results/`: result display, measurements, map controls, CSV/JSON export.
- `reporting/`: evidence charts, claim references, interpretation, report exports.

`frontend/shared/map/` owns reusable viewport behavior and revision-scoped
preferences. Shared navigation styles and icons have their own folders.
Keep date, viewport, selection, and overlay state independent, as described in
the root [AGENTS.md](../AGENTS.md).

## Adding or changing behavior

1. Put domain logic in its owning feature; use shared code only when multiple
   features need the same behavior.
2. Keep external model runtime/storage operations in infrastructure and wire
   dependencies in `nisar/app/cli.py`.
3. Add browser scripts/styles beside their feature. Register public resources
   in `nisar/app/web.py` and include them in the consuming HTML in dependency
   order. Existing asset URLs deliberately remain unchanged.
4. Put Python regression checks in the matching `tests/` feature. Shared fake
   models, synthetic GCOV files, and the ASGI harness live in `tests/support.py`.
   Browser logic checks live in `tests/frontend/`.
5. Keep report schema changes aligned with [REPORT_CONTRACT.md](REPORT_CONTRACT.md).

## Running and compatibility

From the repository root, either command starts the same application:

```bash
python -m nisar --weights models/model.safetensors
python inference.py --weights models/model.safetensors
```

The root `inference.py` is a thin compatibility entry point. Internal imports
use the package locations rather than the former root modules. Weight tools
(`pack`, `convert`, `attach-adapter`) also work through either entry point.

The local bundle is stored at `models/model.safetensors`. With no `--weights`,
the CLI looks there first, then supports an existing root `model.safetensors`
or legacy root `nisar_seg_model_round2.nisar`. Explicit paths remain supported.
Model weights and GCOV inputs are ignored by Git. Temporary uploads and output
images continue to use the existing runtime work directory.

Run regression checks without loading bundled weights:

```bash
python -m unittest discover -s tests -t . -v
node tests/frontend/test_explorer.js
```

The original exported ignore file is retained under `docs/archive/`; the active
ignore rules are in the root `.gitignore`.
