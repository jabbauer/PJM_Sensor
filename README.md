# PJM Sensor

[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/custom-components/hacs)

A Home Assistant integration providing real-time PJM sensor data for monitoring zonal wholesale energy loads, forecasts, and predicting coincident system peaks. This integration leverages PJM's DataMiner 2 API for up-to-date insights.

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

- **Coincident Peak Prediction System/Zone Attribute Description:**
  - **Predicted Peak** – in MW
  - **Predicted Peak Time** – Timestamp of Peak
  - **Observed Peak** – Resets daily
  - **Observed Peak Time** – Timestamp of Peak
  - **Peak Hour Active** – 1, during hour of predicted peak, otherwise 0
    - Use for triggering curtailment on coincident days
  - **High Risk Day** – 1, during day of predicted coincident peak, otherwise 0
    - Use for pre-cooling logic on coincident days
  - **Observed ROC** – System/Zone 2Hr Load Rate of Change - MW/hr
  - **Observed ACC** – System/Zone 2Hr Load Acceleration - MW/hr^2
  - **Forecasted ROC** – System/Zone 2Hr Forecast Rate of Change - MW/hr
  - **Forecasted ACC** – System/Zone 2Hr Forecast Acceleration - MW/hr^2
  - **Bias roc/acc/time** - Future - Error measurement for forecast adjustments
  - **Error History** - Future - Error measurement for forecast adjustments
  - **Top Five Peaks** - Tracks Coincident Peaks (Date/time - MW)

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
   - **Enter your API key** (optional). Without an API key, you may become IP banned by PJM. Utilize for initial testing only.
   - **Choose the sensor entities** you want to enable.
     - By default, **instantaneous_total_load**, **total_short_forecast**, and **total_load_forecast** are selected.
     - Without an API key, only up to 3 sensors can be selected.
     - Per PJM, Non-members may not exceed 6 data connections per minute.

## Bug Fixes - 2.1.5:
- Increased history buffer lengths, 12 to 36
- Lower smoothing constant, 0.3 to 0.2
- Adjusted Kinematic Prediction Thresholds, 50 to 25
- Adjusted logging for debugging purposes
- Additional extra_state_attributes added

## Disclaimer

This integration is not affiliated with, nor supported by, PJM or any PJM member. Use at your own risk.
