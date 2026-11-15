# Change detection pipeline

1. **Load and align the pair** — `nisar/features/imagery/` reads NISAR L2 GCOV geometry and HH/HV power, aligns the common footprint to T0, and identifies jointly valid data.
2. **Prepare the selected area** — `nisar/features/analysis/service.py` averages valid native power cells to the model spacing, converts to dB, and supplies overlapping patches to the detector. The saved training clip ranges and normalization are applied before inference.
3. **Detect** — `nisar/infrastructure/models/change_model.py` runs the same ViT-H encoder on T0 and T1. The CNN fusion, local and refinement branches produce change logits. The detector stitches overlapping predictions, applies the requested threshold, and excludes invalid paired cells.
4. **Measure and render** — The analysis service computes coverage and signed paired HH/HV differences for flagged and other valid cells. It writes clean images and an independently toggleable flag overlay. Reporting builds a deterministic evidence summary with validation limits.
5. **Serve and inspect** — The web app displays the detector results, supports Markdown/plain-text exports, and can answer questions using the active result data. Optional local Qwen chat sees the selected pair and mask only after a user asks a question; it does not run during detection. Saved sample mode reads checksummed artifacts and does not load the detector or HDF5 inputs.

`extract_detector.py` copies the retained tensor bytes from a packed checkpoint into a detector-only safetensors checkpoint, along with the matching configuration and original detector checksum. Only the shared encoder and change head are loaded during live inference.
