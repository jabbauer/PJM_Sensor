# PJM Sensor

[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/custom-components/hacs)

A Home Assistant Integration that enables PJM sensors - useful for monitoring zonal wholesale energy prices, performing Demand Response, or managing capacity charges through detecting coincident system peaks. Calls on PJM's DataMiner 2 API for data access.

Without an API key, users are limited to 6 requests per minute. PJM members are limited to 600 requests per minute.

Note that information and data contained in Data Miner is for internal use only and redistribution of information and or data contained in or derived from Data Miner is strictly prohibited without a PJM membership.

# Installation
1. Under HACS Options, select custom repositories and add
   - Repository: https://github.com/jabbauer/PJM_Sensor/tree/main
   - Type: Integration

3. Search and Download: PJM Sensor for HomeAssistant
4. Restart Home Assistant
5. Under Settings, Devices and Services
   - Add Integration: PJM Sensor
6. Select your utility Zone
7. Select which sensor entities to enable:
   
   - instantaneous_total_load - PJM Instantaneous Load, 5 minute update interval
   - total_short_forecast - PJM 2Hr Forecast, 5 minute update interval
   - total_load_forecast - PJM Daily Forecast
   - instantaneous_zone_load - Zonal Instantaneous Load, 5 minute update interval
   - zone_short_forecast - Zonal 2Hr Forecast, 5 minute update interval
   - zone_load_forecast - Zonal Daily Forecast
   - zonal_lmp - Hourly Average Locational Marginal Price ($/MWh) for selected zone

# Disclaimer
This integration is not affiliated with nor supported by PJM, or any PJM member.
