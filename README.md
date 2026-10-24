# PJM Sensor

**Version 2.3.1** — PJM grid load, load forecasts, coincident-peak predictions, and zonal electricity prices for Home Assistant.

This release targets **Home Assistant 2026.10 or later**. It uses HA's shared HTTP session and validation engine; NumPy is pinned to the version in the HA 2026.10 dependency constraints. Earlier HA releases are not declared supported by this package.

## Installation

Add `https://github.com/jabbauer/PJM_Sensor` as a custom integration repository in HACS, download it, and restart Home Assistant. For a manual install, copy **`custom_components/pjm`** into your HA configuration directory, then restart. The directory must be named `pjm`.

Open **Settings → Devices & services → Add integration → PJM Sensor**. Choose your zone, optional PJM subscription key, and desired sensors. Without your own key, the form limits selection to three sensors. Saved option changes reload the integration automatically.

If upgrading from an old manual install, back up your configuration and remove the obsolete `custom_components/PJM_sensor` code directory after installing the canonical `pjm` directory. Do not delete the integration's configuration entry or `.storage` files. Existing schema-1 entries migrate to schema 2, retaining sensors, zone, thresholds and API key. Schema-2 entries remain schema 2.

## Sensors

- PJM system and zonal instantaneous load, in MW.
- Daily and two-hour load forecasts.
- Real-time zonal LMP, averaged over the requested feed rows, in $/MWh.
- Day-ahead hourly LMP for supported transmission zones, in $/MWh.
- Coincident-peak prediction sensors for the system and configured zone. Their numeric state is observed load; predictions and curtailment status are attributes.

## Peak-hour control

`peak_hour_active` is the curtailment signal. It is evaluated after load updates and by a separate hour-boundary callback using the latest available inputs. A late callback records its actual execution time; it cannot retroactively cover the beginning of an hour. Network, event-loop and downstream automation delays can still affect physical dispatch.

| Cover mode | Policy | Maximum policy span | Default daily elapsed-time cap |
|---|---|---:|---:|
| Secure | Prefer on-hour; mid-hour first activation permitted | First arm's clock hour, with existing lock behavior | 60 minutes |
| Intermediate | Cover-first; normally first arms during minutes 0–6; conditional extension | Up to 2 clock hours | 120 minutes |
| Reliable | Broader cover; less climb waiting; conditional extensions | Up to 3 clock hours | 120 minutes |

The `max_daily_curtailment_minutes` option defaults to **120** and accepts 1–180 minutes. The effective cap is the smaller of that setting and the mode's 60/120/180-minute limit. Reliable can therefore use three hours only if you explicitly raise the daily setting. All active intervals, including lock reactivations, share the same elapsed-time allowance. The cap is per prediction entity, not a combined household or physical-device budget.

Usage and first-arm information are checkpointed to HA storage and carried through reloads. After an interrupted active checkpoint, downtime is conservatively charged before a fresh decision is allowed. Corrupt/unreadable budget storage disables the signal for the remainder of that local day. Accounting resets at local midnight. HA must be running for timers to execute; your device automations remain responsible for device behavior during outages.

Secure remains the setup and migration default. Intermediate is the preferred candidate for the two-hour objective based on our reconstructed-input simulations, not a guarantee of five-event capture. The 2.3.0 replay covered five of five observed proxy peak hours with aligned polling, but only three complete hours with a two-minute poll shift. Do not present those results as official settlement accuracy or measured 2.3.1 production performance.

### Settings and attributes

- Zonal/system peak threshold defaults: **18,000 / 150,000 MW**. Upgrades preserve previously saved thresholds.
- High-risk factor: **0.98 × max(configured threshold, fifth stored peak)** when five peaks are present. A previously seen high-risk condition can remain latched during the day.
- `cover_mode`, `high_risk_day`, `saw_high_risk`, `prediction_time_locked` explain policy state.
- `actual_activation_time` is the actual most recent activation timestamp, separate from the legacy `predicted_peak_time` decision-hour label.
- `curtailment_minutes_used` is usage accounted at the most recent decision; `max_daily_curtailment_minutes` is the effective cap.
- `predicted_peak`, `observed_peak`, `observed_peak_time`, `top_five_peaks`, forecast derivatives and learning diagnostics remain available.
- The five-peak history remains visible through December and resets on **January 1 in the Home Assistant time zone**. If HA is offline then, it catches up on the first update after restart. The reset marker is saved so reloading does not clear the current year's board again.
- `integration_version` is `2.3.1`. Compatible learned-bias data retain their existing internal schema tag.

## Day-ahead hourly prices

The sensor fetches the HA local day's UTC window for the configured zone's mapped pricing node. **`hourly_prices`** is the authoritative list of `{start_utc, start_local, price}` intervals. It retains both repeated autumn hours and handles 23/25-hour days. Numeric attributes `0`–`23` are convenience views in the HA time zone; a missing or repeated hour is `null` rather than an invented average. The current state selects the matching UTC interval.

**If a refresh fails or the current interval is missing, keep the last published price as the best available estimate.** `price_stale` then becomes true; `price_interval_start` and `price_as_of` retain that price's provenance. `as_of` describes the latest fetched curve, which can differ from the retained state's provenance. With no price ever received, the sensor is unavailable until it obtains one. The last numeric value is held in memory; persistence of that price across a Home Assistant restart is not provided.

The new-day curve is fetched once complete, and incomplete curves are retried on subsequent polls. Day-ahead pricing is offered only for mapped supported zones, not the PJM-RTO aggregate. The setup form rejects unsupported combinations.

## Experimental diagnostics

`experimental_coincident_logic` defaults to false. When enabled, a shadow probability model exposes `*_experimental` attributes, persists its own weights/prior, and writes `pjm_pmf_shadow.jsonl` under the HA configuration directory with size-based rotation. It does **not** control production `peak_hour_active`. Keep experimental logging opt-in.

## Development and validation

Run offline regression checks with Python and NumPy installed:

```sh
python -m unittest discover -s tests -v
python -m compileall -q custom_components/pjm
```

The tests execute real source methods with deterministic HA/network adapters. They cover budget accounting, deadlines, restart/reload behavior, stale load, price time zones/DST, retained stale prices, HTTP failures, migration and metadata. They do not replace a clean HA install/upgrade test. The GitHub workflow adds package imports and form-schema checks on the declared minimum HA version; it must pass before a stable release is tagged.

Local research, historical snapshots, household exports and simulation results are excluded from Git by `.gitignore`. Release artifacts contain integration code and documentation, not those local datasets.

See [CHANGELOG.md](CHANGELOG.md) for the 2.3.1 changes.
