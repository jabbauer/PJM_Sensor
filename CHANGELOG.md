# Changelog

## 2.3.1

- Retain the five coincident peaks through December and reset on local January 1, with restart catch-up and a persisted once-per-year marker.
- Base the release on the installed 2.3.0 source, preserving Secure, Intermediate, Reliable, optional shadow diagnostics, and day-ahead prices.
- Evaluate peak-hour policy at scheduled hour boundaries without waiting for HTTP; publish actual activation timestamps separately from decision-hour labels.
- Enforce a configurable daily elapsed-time budget after every policy branch, including locks; persist usage across reloads/restarts and schedule budget expiry. Default maximum is 120 minutes, further limited by the cover mode.
- Reload sensors when saved configuration changes. Preserve the configuration-entry schema and existing options during upgrades; keep Secure as the migration default.
- Replace legacy timeout contexts with bounded, context-managed HTTP requests; reject redirects and HTTP error statuses, and remove unused SciPy and async-timeout dependencies.
- Parse day-ahead prices by UTC interval, preserve repeated DST hours, and use HA-local date boundaries. Keep the last numeric price on failed or incomplete refreshes, with explicit stale/provenance attributes.
- Validate day-ahead zone selection, mask the API-key input, and include English form labels/errors.
- Package the integration under `custom_components/pjm`, correct `cloud_polling`, and target HA 2026.10+ with matching NumPy dependency constraints.
- Add regression checks and a GitHub validation workflow; exclude local research and household datasets from release commits.

2.3.0 and 2.2.0 source snapshots remain available in local ignored analysis directories for reproducible comparisons. No installation is automatically changed by preparing this release.
