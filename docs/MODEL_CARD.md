# Detector evidence and limitations

This page separates what can be recovered from the supplied checkpoint and
inference code from training history and accuracy evidence that are not available
in this repository. It is not an independent accuracy evaluation.

## Checkpoint and inference

The following values were read from the local detector checkpoint metadata.
Architecture and inference behavior can also be inspected in
[`change_model.py`](../nisar/infrastructure/models/change_model.py).

| Property | Recorded value |
| --- | --- |
| Checkpoint format | `nisar_detector_safetensors_v1` |
| Detector format | `nisar_change_v1` |
| Recorded detector SHA-256 identifier | `d11fd2bc10d89fe9150b9ecc7c7478a2766e4b4051b6320ab2504b31b206f506` |
| Architecture | Shared ViT-H encoder; convolutional fusion, local, and refinement branches; head width 128 |
| Inputs | Paired HH/HV backscatter; channels `HHHH`, `HVHV`; frequency A |
| Analysis spacing | 20 metres |
| Input tile size | 224 × 224 cells |
| dB clipping | −50 to +10 dB for both channels |
| Recorded normalization means, HH/HV | −12.89181526634138 / −18.369273772073072 dB |
| Recorded normalization standard deviations, HH/HV | 8.094868909717345 / 7.544678863387776 dB |
| Recorded tile counts | 9,600 train / 4,480 validation |
| Recorded seed | 42 |
| Recorded normalization source fingerprint | `ec65c9b4cb2a2a3557d1a2f893e5f493b82d410570ce70afe4067088a15b9232` |

These are metadata records, not verification of dataset provenance, independent
splits, or performance. The detector identifier is inherited from the checkpoint
configuration; it should not be substituted for a separately computed download
checksum of the extracted weight file.

Inference converts logits to sigmoid scores, combines overlapping tile scores
using a spatial taper, and applies the selected cutoff to jointly valid cells.
The shared mean and absolute-difference features make the architecture invariant
to swapping the two dates. Separate measured After − Before differences supply
the direction of radar change.

## Threshold

The saved sample and live default use **0.10**. This is an operating setting of
the supplied demonstration, not a scientifically validated optimum. A score is
not a calibrated probability of environmental change. There is no documented
threshold sensitivity study or independent evidence supporting a particular
precision/recall tradeoff at this cutoff.

## Intended use and demonstrated behavior

The prototype helps people inspect paired radar observations and prioritize
locations for further investigation. Its saved Karakalpakstan case compares
21 June and 13 September 2026. At the saved cutoff, the detector flagged
999,843 model cells, corresponding to **399.94 km² of projected grid area**.
That is detector-flagged area, not independently confirmed environmental change.

The app preserves spatial selection while switching dates, displays the detector
mask as an independent layer, and exports measurements and provenance. Software
checks establish those implementation behaviors; they do not establish the
detector's accuracy.

## Evidence not established

- The original training code, dataset identity, geographic coverage, label
  definitions, and label-generation or review process are not documented here.
- Recorded train/validation tile counts do not show whether splits are spatially
  independent, whether neighboring/overlapping tiles leak between splits, or
  whether the demonstration landscape was excluded from training.
- No independently verified precision, recall, F1, IoU, or simple-baseline
  comparison is available. Geographic generalization is unestablished.
- Regional Sudochye monitoring is useful external context. Its dates and
  geographic scope differ, and it does not validate individual detector flags.
- A [four-location optical pilot](../evaluation/README.md) now compares exact
  coordinates with nearby Landsat acquisitions. It provides local evidence and
  interpretation failures, with AI-assisted review and explicit quality limits.
  Its selected examples do not supply independent ground-truth labels or
  precision/recall/IoU estimates.
- Unflagged cells are not confirmed unchanged. Optical disagreement alone does
  not establish a radar detection error: the reference must match the physical
  change being evaluated.

## Reproduction status

Saved playback includes the image/result bundle and requires no checkpoint or
source HDF5 products. Live inference additionally requires the detector weights
and paired GCOV products. The repository currently does not provide a public
detector download route or the original training workflow. See the
[`README`](../README.md) for the supported playback and inference commands.

## Evaluation needed

Define a specific target, such as visible surface-water gain/loss, before
labeling. Use independent, spatially matched references near both radar dates;
record dates, resolution, alignment, uncertainty, and label evidence. Assess
flagged and unflagged locations and retain unresolved cases explicitly.

Keep illustrative examples separate from a quantitative evaluation sample.
Freeze evaluation locations and thresholds before inspecting test results.
Compare the learned detector with a simple paired dB-difference baseline on the
same valid cells. Report confusion counts and appropriate metrics for the
labeling unit; use IoU only with suitable spatial reference masks. Any estimates
for the full scene must account for the sampling design. A small local pilot
does not demonstrate generalization to other regions or seasons.
