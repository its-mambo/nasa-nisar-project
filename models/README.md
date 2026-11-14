# Local model artifacts

Place the combined weight bundle at `models/model.safetensors`, the CLI default.
It is ignored by Git. Use `--weights /path/to/bundle` to load another supported
file or model directory. Runtime loading and conversion code lives in
`nisar/infrastructure/models/`.
