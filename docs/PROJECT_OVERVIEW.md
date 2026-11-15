# NISAR SAR Change Explorer

**Project link:** [GitHub repository](https://github.com/kimannavalerevna/nasa-nisar-project)

**One sentence:** Explore paired NISAR radar observations, inspect possible
changes, and choose locations that need independent environmental verification.

## Project overview

NISAR SAR Change Explorer helps environmental analysts and curious visitors
investigate possible landscape change through one shared workflow. Users compare
two radar observations of the same geographic area, toggle a learned detector's
change layer, and inspect measured radar differences with traceable settings and
exports.

The included case study explores Karakalpakstan, Uzbekistan, comparing
21 June and 13 September 2026. A three-stop investigation guides visitors through
stronger radar returns, a mostly unflagged comparison area, and an unresolved
alert. Independent Sudochye wetland monitoring provides regional context while
clearly separating published water measurements from this detector's output.

Four exact locations also have nearby, georeferenced Landsat comparisons:
optical support for a contracting water-like surface, a quiet comparison,
an unsupported water-loss interpretation, and a location blocked by reference
quality. Source crops, local measurements and date gaps are available for review.

The practical decision is where to investigate next. The tool helps users select
locations and evidence to check, while keeping measured radar change separate
from hypotheses about water, vegetation, or physical causes.

A saved analysis, images, and reports are included for local exploration without
a GPU, detector weights, or source HDF5 downloads. Live inference supports paired
GCOV products when the detector checkpoint is supplied. Optional local Qwen chat
uses result imagery and measured metadata.

The current prototype has not independently established detection accuracy.
Flagged area is an investigation lead, not confirmed environmental change.

## Supporting links

- [Run the included case study](../README.md#explore-the-saved-sample-start-here)
- [Detector metadata, threshold, and evidence limits](MODEL_CARD.md)
- [Location-matched optical pilot and reproducible evidence](../evaluation/README.md)
- [Pipeline architecture](ARCHITECTURE.md)
- [Saved sample details and troubleshooting](../samples/README.md)

The repository link is a source-code and local-demo entry point. It is not a
publicly hosted running app, and the detector weights are not included in Git.
