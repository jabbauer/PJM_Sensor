# PJM Sensor

[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/custom-components/hacs)

A Home Assistant integration that seductively provides PJM sensor data for monitoring zonal wholesale energy loads, forecasts, and even predicting coincident system peaks. This integration taps into PJM's DataMiner 2 API for real-time insights.

## Features

- **Optional API Key:**  
  - **With API Key:** Unlock the full array of sensor options with higher rate limits (up to 600 requests per minute for PJM members) and enable as many sensors as you desire.  
  - **Without API Key:** The integration will automatically fetch the subscription key from PJM and restrict you to a maximum of **3 sensor entities** with a lower rate limit (approximately 6 requests per minute).

- **Default Sensors:**  
  By default, only these three alluring sensors are enabled:  
  - **instantaneous_total_load** – PJM Instantaneous Load (updates every 5 minutes)  
  - **total_short_forecast** – PJM 2Hr Forecast (updates every 5 minutes)  
  - **total_load_forecast** – PJM Daily Forecast  

- **Additional Sensor Options (API key required):**  
  When you provide your API key, you can enable extra sensors, including:  
  - **instantaneous_zone_load** – Zonal Instantaneous Load (5 minute update interval)  
  - **zone_short_forecast** – Zonal 2Hr Forecast (5 minute update interval)  
  - **zone_load_forecast** – Zonal Daily Forecast  
  - **zonal_lmp** – Hourly Average Locational Marginal Price (LMP) for the selected zone  
  - **coincident_peak_prediction_zone** – Coincident Peak Prediction (Zone)  
  - **coincident_peak_prediction_system** – Coincident Peak Prediction (System)

- **Rate Limiting:**  
  Without an API key, you're limited to a modest request rate (~6 requests per minute) to play nicely with PJM’s restrictions. Provide an API key, and you can indulge in up to 600 requests per minute if you're a PJM member.

> **Note:** Data provided by PJM's DataMiner 2 API is for internal use only. Redistribution of this data or any derivative information is strictly prohibited unless you are a PJM member.

## Installation

1. In HACS, go to **HACS > Integrations**.
2. Click on **+ Explore & add custom repository**.
3. Add the repository with the following details:
   - **Repository:** `https://github.com/jabbauer/PJM_Sensor`
   - **Category:** Integration
4. Once added, search for **PJM Sensor** and install it.
5. Restart Home Assistant.
6. Navigate to **Settings > Devices & Services** and add the **PJM Sensor** integration.
7. In the configuration flow:
   - **Select your utility zone.**
   - **Enter your API key** (if you have one). If left blank, the integration will fetch the subscription key and restrict you to a maximum of **3 sensor entities**.
   - **Choose the sensor entities** you wish to enable.
     - By default, only **instantaneous_total_load**, **total_short_forecast**, and **total_load_forecast** are selected.
     - Without an API key, you may only select up to 3 sensors in total.

## Disclaimer

This integration is not affiliated with, nor supported by, PJM or any PJM member.
