# PJM Sensor

[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/custom-components/hacs)

A Home Assistant Integration that enables PJM sensors - useful for monitoring zonal wholesale energy prices, performing Demand Response, or managing capacity charges through detecting coincident system peaks.

Without an API key, users are limited to 6 requests per minute. PJM members are limited to 600 requests per minute.

Users can request an API Key from PJM by following the guide below:
https://www.pjm.com/-/media/etools/data-miner-2/data-miner-2-api-guide.ashx

Note that information and data contained in Data Miner is for internal use only and redistribution of information and or data contained in or derived from Data Miner is strictly prohibited without a PJM membership.

# Configuration:
1. Under HACS Options, select custom repositories and add
   - Repository: https://github.com/jabbauer/PJM_Sensor/tree/main
   - Type: Integration

3. Search and Download: PJM Sensor for HomeAssistant
4. Restart Home Assistant
5. Under Settings, Devices and Services
   - Add Integration: PJM Sensor
6. Select your utility Zone
7. Select which sensor entities to enable:
   - Zonal Instantaneous Load - current zonal load
   - System Instantaneous Load - current PJM systemwide load
   - Zonal Daily Forecast - Peak forecast for selected zone for day
   - System Daily Forecast - Peak forecast for PJM system for day
   - Zonal 2Hr Forecast - Peak forecast for selected Zone in future 2Hr window
   - System 2Hr Forecast - Peak forecast for PJM system in future 2Hr window
   - Zonal LMP - Locational Marginal Price ($/MWh) for selected zone
