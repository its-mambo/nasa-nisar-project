# Evidence report, chat, and research context

The current feature is the structured, exportable **Evidence & interpretation**
report. Its contract is `nisar_evidence_report_v1` in
[`nisar/features/reporting/evidence_report.py`](../nisar/features/reporting/evidence_report.py).

## Claim boundaries

- **Measured:** application-calculated backscatter and valid-data statistics.
- **Detector output:** model flags, thresholds, and areas derived from flags.
- **AI observation:** an unverified description of supplied imagery/evidence.
- **Hypothesis:** an unconfirmed physical explanation with explicit assumptions.
- **Next check:** a recommended independent source and a result that would
  support or weaken the hypothesis. Recommended sources have not been consulted.
- **External regional context:** a separately reviewed source snapshot, with
  its dates, geographic scope, and limitations. It does not validate detector flags.

The application supplies all numeric values and chart data. Qwen returns a JSON
object with `observations`, one optional `hypothesis`, and `verification`.
Each claim cites known evidence IDs; observations can also be cited by a
hypothesis. The prompt in
[`qwen_evidence_prompt.py`](../nisar/features/reporting/qwen_evidence_prompt.py) defines exact fields and
limits. The validator rejects unknown references, repeated observation text,
generated numeric measurements/confidence percentages, and unverified URLs.
It checks syntax and provenance references, not scientific correctness.

No flags or acquisition warnings leave physical attribution open. Malformed,
legacy plain-text, or otherwise unusable model output does not remove the
measured report. It stays available in the technical inspector; the main report
clearly states that a structured interpretation was unavailable.

## Saved-sample plain-text fallback

The saved-sample builder explicitly opts into a presentation fallback when
Qwen produces usable prose or misses JSON fields. `ai.status: text_fallback`
contains `text` and `presentation_note`: original model sentences shown as
unverified plain text, without fabricated uncertainty fields or validated
evidence chips. `ai.observations` stays empty, `ai.hypothesis` stays null, and
physical attribution remains unresolved. Numeric-measurement and URL checks
still apply. Broken JSON and empty responses do not produce an explanation.
Acquisition warnings or zero flags suppress the JSON hypothesis in this view.

The browser uses text nodes, and Markdown/plain-text exports include the same
explanation and disclosure. The strict live-run validator remains unchanged.
`cache_sample.py --refresh-report` republishes the saved report with checksums
and a parent revision; original model output, measurements, and images are
preserved. This is a display fallback, not a new inference or model repair.

## Model evaluation before enabling chat

The existing decoder adapter was used with a plain-text prompt. Changing the
prompt to JSON and allowing a longer response does not prove that the adapted
model will reliably follow the contract. Test on representative real pairs:

1. Flagged regions with a measured contrast against other valid cells.
2. Opposite HH/HV directions, broad background shifts, and sparse valid data.
3. No flags, insufficient samples, reversed dates, and acquisition warnings.
4. Unsupported feature names or mechanisms, missing references, repeated
   claims, invented numbers, and malformed or truncated JSON.

Audit raw responses in the existing decoder trace. No physical-cause accuracy
claim follows from format validation. Automated checks use synthetic GCOV data
and model stubs; the full adapted Qwen model still needs this evaluation.

## Future chat

Attach a conversation to one immutable result revision, its exact input
panels, evidence catalog, and report. Replies use the same claim categories
and evidence references. Retain the report as a stable reference and keep chat
an optional follow-up panel. Avoid rerunning detection just to answer a
question about existing measurements. Source-dependent questions should name
the missing evidence rather than inventing an answer.

## Future web research / scraper

The Karakalpakstan saved sample now includes a reviewed context record in
`nisar/features/samples/karakalpakstan_context.json`, attached by `context.py`
only when the saved sample's dates, ROI, and CRS match. This is a curated
snapshot, not a live retrieval service. It adds `evidence_report.external_context`
to playback responses and exports without rewriting the immutable inference
files, raw Qwen output, measured values, or source checksums. The report's
validation note distinguishes the original run from subsequent context review.

Unchanged JPEG map excerpts are bundled in
`frontend/features/results/reference/`; full source PDFs are linked at the publisher.
The June map is Figure 1 on PDF page 2;
the September map is Figure 2 on PDF page 4. The `pdfimages -j` extraction
preserves embedded JPEG bytes. Sudochye values are from the September report's
Table 3 (PDF page 5), with the June value cross-checked against the June report's
Table 2 (PDF page 3). PDF and image SHA-256 values identify the reviewed snapshots.
Dates are observation dates, not assumed publication dates. Hectares are divided
by 100 for km². Signed day offsets are relative to the nearest radar acquisition.

The published maps cover different regional extents and are not registered to
the SAR grid. Blue areas denote the publisher's water classification, not the
app's change mask. No patch-level optical validation has been completed, and
recommended AI verification steps remain unconsulted. The context record is
excluded from the AI's measured-evidence catalog and is never treated as a
calibration or accuracy estimate.

Chat and web retrieval are not enabled by this change. A retrieval adapter can
later supply **external context** records containing a stable source ID, URL,
title, publisher, publication/observation date, retrieval timestamp, geographic
scope, and the supporting excerpt. Distinguish a recommended source from an
actually retrieved source. Check its relevance to the selection and both
acquisition dates before citing it.

Retrieved text is source content, not instructions. It must not modify the
report contract or request unrelated actions. Cite only retrieved records,
show them in a separate external-context category, and include their provenance
in exports. A news page or contextual dataset alone does not validate the
physical cause of a particular radar patch.

## Exports and figures

`GET /report.md?v=REV` and `GET /report.txt?v=REV` export the same classified
claims and exact evidence as the app, with scope, timestamps, threshold,
projection, and checkpoint identifier when recorded. Text exports represent
the chart values in portable text, without ephemeral local image URLs.

The report's coverage donut uses mutually exclusive model-grid categories:
flagged, valid without flags, and without paired model data. Its percentage
denominator is the entire model grid. The detector's flagged percentage uses
only jointly valid cells. Signed HH/HV bars show paired medians in a leading
sector, rather than implying a trend from only two dates.
