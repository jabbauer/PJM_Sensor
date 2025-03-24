# PJM Sensor - Version 2.1.1

[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/custom-components/hacs)

A Home Assistant integration providing real-time PJM sensor data for monitoring zonal wholesale energy loads, forecasts, and predicting coincident system peaks. This integration leverages PJM's DataMiner 2 API for up-to-date insights.

## What's New in Version 2.1.1

- **Persistent Storage for Peak History:**
  - Introduced persistent storage for the top five historical peaks, ensuring data continuity across Home Assistant restarts.
  - Peak history resets annually on October 1st, aligning with PJM's planning year.
 
- **Refined Prediction Algorithm:**
  - Implemented a weighted-average approach combining daily forecasts, short-term forecasts, and real-time kinematic predictions to improve the accuracy of peak forecasts.
  - Enhanced logic for smoothing the observed rate-of-change (ROC) and acceleration (ACC) values, reducing noise in derivative calculations.

- **Enhanced API Rate Limiting & Retries:**
  - Improved PJM API interaction with exponential backoff strategies, ensuring robust handling of API rate limits and network errors.

## Bug Fixes:
- Fixed issues where the predicted peak could be incorrectly set to past times.
- Addressed potential calculation errors in derivative smoothing and kinematic prediction methods.
- Corrected handling of API responses to prevent repeated errors in case of API rate limits.

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
