# PJM Sensor

[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg?style=for-the-badge)](https://github.com/custom-components/hacs)

A Home Assistant integration providing real-time PJM sensor data for monitoring zonal wholesale energy loads, forecasts, and predicting coincident system peaks. This integration leverages PJM's DataMiner 2 API for up-to-date insights.

---

## Features

This integration uses PJM’s DataMiner 2 API to provide live and forecasted insight into grid conditions. It supports both basic monitoring and advanced predictive capabilities.

### 🔄 Real-Time & Forecast Load Sensors

- **System Load**: PJM-wide real-time load (updated every 5 minutes)
- **System Forecasts**:
  - 2-hour forecast (updated every 5 minutes)
  - Daily forecast (updated hourly)
- **Zonal Load & Forecasts**: Instantaneous and forecasted values for your selected PJM zone
- **Zonal LMP**: Hourly average Locational Marginal Price (LMP) for your zone

### 🔮 Coincident Peak Prediction (System & Zone)

- Predicts daily system or zonal peak load time and magnitude using:
  - Real-time derivative analysis (rate of change, acceleration)
  - 2-hour and daily forecasts
  - Kinematic modeling (quadratic) with adaptive time and magnitude biasing
- Tracks and stores the **top five daily peaks** across the season
- Designed to support **5CP Capacity PLC and Network Service PLC management** or load curtailment strategies

#### Exposed Attributes:
| Attribute | Description |
|----------|-------------|
| `predicted_peak` | Forecasted load in MW |
| `predicted_peak_time` | Timestamp of predicted peak |
| `observed_peak` | Highest observed real-time load today |
| `observed_peak_time` | Timestamp of observed peak |
| `peak_hour_active` | `true` during the predicted peak hour |
| `high_risk_day` | `true` if forecasted peak ≥ 95% of the higher of: the 5th highest historical peak or a configured threshold |
| `observed_roc` / `observed_acc` | Real-time load rate-of-change and acceleration |
| `forecasted_roc` / `forecasted_acc` | Forecasted load rate-of-change and acceleration |
| `bias_roc`, `bias_acc`, `time_bias`, `magnitude_bias` | Adaptive corrections learned from previous prediction errors |
| `error_history` | Recent magnitude error history (MW) |
| `top_five_peaks` | List of the top five daily peaks observed this season |

> All timestamps are in ISO 8601 format and localized to your Home Assistant instance's time zone.

---

### ⚙️ Configuration Settings

- `peak_threshold_zone` / `peak_threshold_system`  
  Minimum load (in MW) for triggering high-risk day logic. Default: 17,000 (zone), 138,000 (system)

- `accuracy_threshold`  
  *(Reserved for future use.)* Intended for evaluating forecast confidence before triggering peak alerts.

---

### 🔑 API Key Behavior

- **With API Key**:
  - Full access to all sensors
  - Non-members may not exceed 6 data connections per minute
- **Without API Key**:
  - Limited to 3 sensor entities
  - For evaluation or testing only
  - Risk of IP ban

---

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

## 💡 Example: Load Curtailment Automation

```yaml
- alias: Curtail HVAC During PJM Peak
  trigger:
    - platform: state
      entity_id: sensor.coincident_peak_prediction_system
      attribute: peak_hour_active
      to: 'true'
  action:
    - service: climate.set_temperature
      target:
        entity_id: climate.house
      data:
        temperature: 78
```


## Changelog - 2.1.5:
- Increased history buffer lengths, 12 to 36
- Lower smoothing constant, 0.3 to 0.2
- Adjusted Kinematic Prediction Thresholds, 50 to 25
- Adjusted logging for debugging purposes
- Additional extra_state_attributes added

## Disclaimer

This integration is not affiliated with, nor supported by, PJM or any PJM member. Use at your own risk. Data from PJM's DataMiner 2 API is for internal use only. Redistribution of this data or any derivative information is prohibited unless you are a PJM member.

