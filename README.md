# PJM Sensor

[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/custom-components/hacs)

A Home Assistant integration providing real-time PJM sensor data for monitoring zonal wholesale energy loads, forecasts, and predicting coincident system peaks. This integration utilizes PJM's DataMiner 2 API for comprehensive insights into the grid.

## Features

- **API Key Support:**  
  - **With API Key:** Unlocks full sensor capabilities with higher rate limits (up to 600 requests per minute for PJM members) and no sensor limit.  
  - **Without API Key:** Automatically retrieves the subscription key from PJM but limits you to a maximum of **3 sensor entities** and a lower rate limit (~6 requests per minute).

- **Default Sensors:**  
  Enabled by default:
  - **PJM System Load** – Instantaneous system load (updates every 5 minutes).
  - **PJM 2Hr Forecast** – Short-term load forecast with attributes including forecast_hour_ending (updates every 5 minutes).
  - **PJM Daily System Forecast** – Daily peak load forecast with forecast_hour_ending attribute (updates hourly).

- **Additional Sensor Options (API Key Required):**  
  Unlock additional sensors:
  - **instantaneous_zone_load** – Zonal Instantaneous Load (5-minute updates)
  - **zone_short_forecast** – Zonal 2Hr Forecast (5-minute updates)
  - **zone_load_forecast** – Zonal Daily Forecast
  - **zonal_lmp** – Hourly Average Locational Marginal Price (LMP) per zone
  - **coincident_peak_prediction_zone** – Coincident Peak Prediction (Zone) using real-time trends and regression analysis
  - **coincident_peak_prediction_system** – Coincident Peak Prediction (System) with 5CP logic

- **Advanced Coincident Peak Prediction:**  
  - Uses real-time load trends, rate-of-change analysis, and quadratic regression to estimate system coincident peaks.
  - Automatically tracks historical peak data to refine predictions.
  - Identifies **high-risk days** for 5CP billing considerations.

- **Rate Limiting:**  
  Without an API key, request rates are limited to **6 requests per minute**. With an API key, PJM members can make up to **600 requests per minute**.

> **Note:** Data from PJM's DataMiner 2 API is for internal use only. Redistribution or derivative data sharing is prohibited unless you are a PJM member.

## Installation

1. In HACS, go to **HACS > Integrations**.
2. Click **+ Explore & add custom repository**.
3. Add the repository:
   - **Repository:** `https://github.com/jabbauer/PJM_Sensor`
   - **Category:** Integration
4. Search for **PJM Sensor** and install it.
5. Restart Home Assistant.
6. Navigate to **Settings > Devices & Services** and add the **PJM Sensor** integration.
7. Configuration steps:
   - Select your **utility zone**.
   - Enter your **API key** (optional). Without it, you're limited to **3 sensor entities**.
   - Choose the sensors you want to enable.
     - Default: **instantaneous_total_load**, **total_short_forecast**, and **total_load_forecast**.
     - Without an API key, a maximum of **3 sensors** can be selected.

## Disclaimer

This integration is not affiliated with, nor officially supported by, PJM or any PJM member.

