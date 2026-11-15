# Evidence report contract

Analysis results expose `evidence_report` with `schema_version: "nisar_evidence_report_v2"`. The report draws from result measurements and contains:

- `finding`: detector summary and references into the evidence catalog.
- `evidence`: detector flag counts/area, coverage, signed sector medians and image references. Each row includes its source paths.
- `featured_evidence`: IDs to highlight in the evidence panel.
- `caution`, `validation`: limitations, including warnings for unsuitable acquisition comparisons.
- `checks`: suggested independent comparisons. They have not been performed by the app.
- `external_context`: reviewed regional source records when available for the saved sample. This material does not validate particular detector flags.

HH/HV deltas are medians of paired After-minus-Before differences on the detector grid. Compare flagged and other valid cells in the same sector. Detector flags do not identify a physical cause. No scene class, attribution, or generated explanation is emitted. Old saved samples are projected onto this report contract when loaded; original files and checksums are unchanged.

The optional Results chat streams on `POST /chat`. It uses read-only snapshots of the completed result and never modifies the detector report. Saved samples supply stored before/after images and the visible footprint of their overlay, which may omit faint flags.
