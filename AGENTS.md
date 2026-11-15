# Project context

Build NISAR SAR Change Explorer for two equally important personas:

- **Environmental analyst:** needs precise spatial selection, comparable dates,
  traceable thresholds and measurements, useful exports, and clear uncertainty
  when assessing possible environmental change.
- **Enthusiast:** wants to explore before/after imagery and understand detected
  changes without prior knowledge of SAR, model grids, or coordinate systems.

These are design personas, not authentication or permission roles. Keep one
shared workflow. Use plain-language labels and explanations by default, with
technical details available through progressive disclosure. Explain what a
setting changes and why it matters. Distinguish measured radar change from AI
interpretation and physical-cause claims.

Keep dates, zoom/pan, selection, and overlay state independent. Changing an
image date must preserve the selected geographic area. A change mask is an
on/off layer over either date, not a third acquisition or separate image mode.
Show model progress in a dedicated, compact view rather than below the setup
sidebar.
