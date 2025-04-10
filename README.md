# PJM Sensor

[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/custom-components/hacs)

A Home Assistant integration providing real-time PJM sensor data for monitoring zonal wholesale energy loads, forecasts, and predicting coincident system peaks. This integration leverages PJM's DataMiner 2 API for up-to-date insights.

## Bug Fixes - 2.1.4:
- Update to High Risk Day logic to use maximum of 5th highest peak or user configured peak_threshold: Default PJM - 138,000MW, Zone - 17,000MW
- Adjusted def extra_state_attributes to correctly format top five peaks

## Features

- **Optional API Key:**
  - **With API Key:** Gain access to the full suite of sensors.
  - **Without API Key:** The integration will fetch the subscription key from PJM but limit you to a maximum of **3 sensor entities** with a lower rate limit (Recommended for testing only, otherwise risk IP ban).

- **Default Sensors:**
  By default, the following three sensors are enabled:
  - **instantaneous_total_load** – PJM Instantaneous Load (updates every 5 minutes)
  - **total_short_forecast** – PJM 2Hr Forecast (updates every 5 minutes)
  - **total_load_forecast** – PJM Daily Forecast

- **Additional Sensor Options (API key required):**
  When an API key is provided, additional sensors can be enabled, including:
  - **instantaneous_zone_load** – Zonal Instantaneous Load (updates every 5 minutes)
  - **zone_short_forecast** – Zonal 2Hr Forecast (updates every 5 minutes)
  - **zone_load_forecast** – Zonal Daily Forecast
  - **zonal_lmp** – Hourly Average Locational Marginal Price (LMP) for the selected zone
  - **coincident_peak_prediction_zone** – Coincident Peak Prediction (Zone)
  - **coincident_peak_prediction_system** – Coincident Peak Prediction (System)

> **Note:** Data from PJM's DataMiner 2 API is for internal use only. Redistribution of this data or any derivative information is prohibited unless you are a PJM member.

## Installation

1. In HACS, go to **HACS > Integrations**.
2. Click on **+ Explore & add custom repository**.
3. Add the repository:
   - **URL:** `https://github.com/jabbauer/PJM_Sensor`
   - **Category:** Integration
4. Search for **PJM Sensor** and install it.
5. Restart Home Assistant.
6. Navigate to **Settings > Devices & Services** and add the **PJM Sensor** integration.
7. In the configuration flow:
   - **Select your utility zone.**
   - **Enter your API key** (optional). Without an API key, the integration fetches the subscription key but limits you to 3 sensors.
   - **Choose the sensor entities** you want to enable.
     - By default, **instantaneous_total_load**, **total_short_forecast**, and **total_load_forecast** are selected.
     - Without an API key, only up to 3 sensors can be selected.

## Disclaimer

This integration is not affiliated with, nor supported by, PJM or any PJM member. Use at your own risk.
