"""
PJM Sensor Integration
----------------------

This module provides support for multiple PJM sensors—including the brand
new Coincident Peak Prediction sensor that uses real-time load trends,
derivative analysis, and piecewise quadratic regression to predict coincident
peaks at the start of the hour. API calls will use your provided API key if
available; otherwise, they'll fall back to fetching the subscription key.
"""

import asyncio
from collections import deque, defaultdict
from datetime import datetime, date, time, timezone, timedelta
import logging
import urllib.parse
import time as time_module

import async_timeout
import aiohttp
import numpy as np
from scipy.optimize import curve_fit

from homeassistant.components.sensor import SensorEntity
from homeassistant.components.sensor import SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.storage import Store
from homeassistant.util import Throttle
from homeassistant.util import dt as dt_util
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    DOMAIN,
    CONF_API_KEY,
    CONF_INSTANTANEOUS_ZONE_LOAD,
    CONF_INSTANTANEOUS_TOTAL_LOAD,
    CONF_ZONE_LOAD_FORECAST,
    CONF_TOTAL_LOAD_FORECAST,
    CONF_ZONE_SHORT_FORECAST,
    CONF_TOTAL_SHORT_FORECAST,
    CONF_ZONAL_LMP,
    CONF_COINCIDENT_PEAK_PREDICTION_ZONE,
    CONF_COINCIDENT_PEAK_PREDICTION_SYSTEM,
    CONF_PEAK_THRESHOLD,
    CONF_ACCURACY_THRESHOLD,
    DEFAULT_PEAK_THRESHOLD_ZONE,
    DEFAULT_PEAK_THRESHOLD_SYSTEM,
    DEFAULT_ACCURACY_THRESHOLD,
    ZONE_TO_PNODE_ID,
    SENSOR_TYPES,
)

_LOGGER = logging.getLogger(__name__)

# Define resource URLs
RESOURCE_INSTANTANEOUS = 'https://api.pjm.com/api/v1/inst_load'
RESOURCE_FORECAST = 'https://api.pjm.com/api/v1/load_frcstd_7_day'
RESOURCE_SHORT_FORECAST = 'https://api.pjm.com/api/v1/very_short_load_frcst'
RESOURCE_LMP = 'https://api.pjm.com/api/v1/rt_unverified_fivemin_lmps'
RESOURCE_SUBSCRIPTION_KEY = 'https://dataminer2.pjm.com/config/settings.json'

MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS = timedelta(seconds=300)  # 5 minutes for load, LMPs
MIN_TIME_BETWEEN_UPDATES_FORECAST = timedelta(seconds=3600)  # 1 hour for forecasts

PJM_RTO_ZONE = "PJM RTO"
FORECAST_COMBINED_ZONE = 'RTO_COMBINED'
MAX_HISTORY_SIZE = 300  # about 25 hours of data at 5-min intervals

# Standard quadratic function
def _quadratic(x, a, b, c):
    return a * x**2 + b * x + c

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback):
    """Set up the PJM sensor platform from a config entry."""
    zone = entry.data["zone"]
    selected_sensors = entry.data["sensors"]
    pjm_data = PJMData(async_get_clientsession(hass), entry.data.get(CONF_API_KEY))
    dev = []

    for sensor_type in selected_sensors:
        identifier = zone
        if sensor_type == CONF_INSTANTANEOUS_TOTAL_LOAD:
            identifier = PJM_RTO_ZONE
        if sensor_type in (CONF_TOTAL_LOAD_FORECAST, CONF_TOTAL_SHORT_FORECAST):
            identifier = FORECAST_COMBINED_ZONE

        if sensor_type == CONF_ZONAL_LMP:
            pnode_id = ZONE_TO_PNODE_ID.get(zone)
            if pnode_id is None:
                _LOGGER.error("Invalid zone provided for LMP: %s", zone)
                continue
            dev.append(PJMSensor(pjm_data, sensor_type, pnode_id, None))
        elif sensor_type in (CONF_COINCIDENT_PEAK_PREDICTION_ZONE, CONF_COINCIDENT_PEAK_PREDICTION_SYSTEM):
            if sensor_type == CONF_COINCIDENT_PEAK_PREDICTION_ZONE:
                peak_threshold = entry.data.get("peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE)
            else:
                peak_threshold = entry.data.get("peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM)
            accuracy_threshold = entry.data.get(CONF_ACCURACY_THRESHOLD, DEFAULT_ACCURACY_THRESHOLD)
            dev.append(CoincidentPeakPredictionSensor(
                pjm_data, zone if sensor_type == CONF_COINCIDENT_PEAK_PREDICTION_ZONE else PJM_RTO_ZONE,
                peak_threshold, accuracy_threshold, sensor_type, hass))
        else:
            dev.append(PJMSensor(pjm_data, sensor_type, identifier, None))

    async_add_entities(dev, True)

    for index, entity in enumerate(dev):
        delay = 12 + (index * 12)
        hass.async_create_task(schedule_delayed_update(entity, delay))

async def schedule_delayed_update(entity, delay):
    """Schedule an update after a delay using async sleep."""
    await asyncio.sleep(delay)
    await entity.async_update()

class PJMSensor(SensorEntity):
    """Implementation of a standard PJM sensor."""
    def __init__(self, pjm_data, sensor_type, identifier, name):
        super().__init__()
        self._pjm_data = pjm_data
        self._type = sensor_type
        self._identifier = identifier
        self._unit_of_measurement = SENSOR_TYPES[sensor_type][1]
        self._attr_unique_id = f"pjm_{sensor_type}_{identifier}"
        self._state = None
        self._forecast_data = None

        if name:
            self._attr_name = name
        else:
            self._attr_name = SENSOR_TYPES[sensor_type][0]
            if sensor_type in (CONF_INSTANTANEOUS_ZONE_LOAD, CONF_ZONE_LOAD_FORECAST, CONF_ZONE_SHORT_FORECAST):
                self._attr_name = f'{identifier} {SENSOR_TYPES[sensor_type][0]}'
            elif sensor_type == CONF_ZONAL_LMP:
                zone_name = next((zone for zone, pid in ZONE_TO_PNODE_ID.items() if pid == identifier), None)
                if zone_name:
                    self._attr_name = f'{zone_name} {SENSOR_TYPES[sensor_type][0]}'
                else:
                    self._attr_name += ' ' + f'{identifier}'
        # Enable long-term statistics for system and zone load or LMP
        if sensor_type in (CONF_INSTANTANEOUS_ZONE_LOAD, CONF_INSTANTANEOUS_TOTAL_LOAD, CONF_ZONAL_LMP):
            self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name(self):
        return self._attr_name

    @property
    def unique_id(self):
        return self._attr_unique_id

    @property
    def icon(self):
        if self._type in [
            CONF_ZONE_LOAD_FORECAST,
            CONF_TOTAL_LOAD_FORECAST,
            CONF_ZONE_SHORT_FORECAST,
            CONF_TOTAL_SHORT_FORECAST
        ]:
            return "mdi:chart-timeline-variant"
        elif self._type in [
            CONF_INSTANTANEOUS_ZONE_LOAD,
            CONF_INSTANTANEOUS_TOTAL_LOAD
        ]:
            return "mdi:transmission-tower-export"
        elif self._type == CONF_ZONAL_LMP:
            return "mdi:meter-electric"
        else:
            return "mdi:flash"

    @property
    def unit_of_measurement(self):
        return self._unit_of_measurement

    @property
    def native_value(self):
        return self._state

    @property
    def extra_state_attributes(self):
        attr = {}
        if self._identifier and self._type not in [CONF_TOTAL_LOAD_FORECAST, CONF_TOTAL_SHORT_FORECAST]:
            attr["identifier"] = self._identifier
            
        if self._type in [CONF_INSTANTANEOUS_ZONE_LOAD, CONF_INSTANTANEOUS_TOTAL_LOAD]:
            attr["observed_rate_of_change"] = self._observed_roc

        if self._type in [CONF_TOTAL_LOAD_FORECAST, CONF_ZONE_LOAD_FORECAST]:
            attr["forecast_hour_ending"] = self._forecast_hour_ending.isoformat() if hasattr(self, "_forecast_hour_ending") and self._forecast_hour_ending else None

        if self._type in [CONF_TOTAL_SHORT_FORECAST, CONF_ZONE_SHORT_FORECAST]:
            attr["forecast_peak_time"] = self._forecast_hour_ending.isoformat() if hasattr(self, "_forecast_hour_ending") and self._forecast_hour_ending else None
            attr["forecast_rate_of_change"] = self._forecast_roc
            # attr["forecast_data"] = self._forecast_data
        return attr

    async def async_update(self):
        try:
            if self._type in (CONF_INSTANTANEOUS_ZONE_LOAD, CONF_INSTANTANEOUS_TOTAL_LOAD):
                await self.update_load()
            elif self._type == CONF_ZONAL_LMP:
                await self.update_lmp()
            elif self._type in (CONF_TOTAL_SHORT_FORECAST, CONF_ZONE_SHORT_FORECAST):
                await self.update_short_forecast()
            else:
                await self.update_forecast()
        except Exception as err:
            _LOGGER.error("Update failed: %s", err)

    @Throttle(MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS)
    async def update_load(self):
        load = await self._pjm_data.async_update_instantaneous(self._identifier)
        if load is not None:
            self._state = load

        # 2) Append to a rolling history
        now_utc = datetime.now(timezone.utc)
        if not hasattr(self, "_load_history"):
            self._load_history = deque(maxlen=12)  # ~ 1 hour if each update is 5 min
        self._load_history.append((now_utc, load))

        # 3) Compute derivative over this 1-hour window
        self._observed_roc = self._compute_instantaneous_roc()

    @Throttle(MIN_TIME_BETWEEN_UPDATES_FORECAST)
    async def update_forecast(self):
        forecast_data = await self._pjm_data.async_update_forecast(self._identifier)
        if forecast_data is not None:
            max_forecast = max(forecast_data, key=lambda x: x["forecast_load_mw"])
            peak_forecast_load = max_forecast["forecast_load_mw"]
            self._state = peak_forecast_load
            self._forecast_hour_ending = max_forecast["forecast_hour_ending"]


    @Throttle(MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS)
    async def update_short_forecast(self):
        forecast_data = await self._pjm_data.async_update_short_forecast(self._identifier)
        if forecast_data and len(forecast_data) > 1:
            #self._forecast_data = forecast_data
            # 1) Compute the maximum forecast load & set state
            max_item = max(forecast_data, key=lambda x: x["forecast_load_mw"])
            self._state = max_item["forecast_load_mw"]
            self._forecast_hour_ending = max_item["forecast_hour_ending"]
            # 2) Compute the derivative (MW/hr) for the chosen window
            self._forecast_roc = self._compute_forecast_rate_of_change(forecast_data)
        else:
            # No valid data
            #self._forecast_data = None
            self._forecast_roc = 0

    @Throttle(MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS)
    async def update_lmp(self):
        lmp = await self._pjm_data.async_update_lmp(self._identifier)
        if lmp is not None:
            self._state = lmp

    def _compute_instantaneous_roc(self):
        """Compute MW/hr slope from the oldest to newest in _load_history."""
        if not hasattr(self, "_load_history") or len(self._load_history) < 2:
            return 0
        oldest_time, oldest_val = self._load_history[0]
        newest_time, newest_val = self._load_history[-1]
        delta_load = newest_val - oldest_val
        delta_time = (newest_time - oldest_time).total_seconds() / 3600
        if delta_time <= 0:
            return 0
        return delta_load / delta_time

    def _compute_forecast_rate_of_change(self, data):
        """
        Example approach:
        - We'll calculate a slope over the next 30 minutes from data[0] to data that ends by +30min
        - Could also do entire 2 hours, or up to the peak, etc.
        """
        if not data or len(data) < 2:
            return 0

        # Filter data for next 30 minutes from the first forecast
        start_time = data[0]["forecast_hour_ending"]
        window_end_time = start_time + timedelta(minutes=30)
        segment = [x for x in data if x["forecast_hour_ending"] <= window_end_time]
        if len(segment) < 2:
            # fallback: just use entire 2-hour window
            segment = data

        start = segment[0]
        end = segment[-1]
        delta_load = end["forecast_load_mw"] - start["forecast_load_mw"]
        delta_time_hrs = (end["forecast_hour_ending"] - start["forecast_hour_ending"]).total_seconds() / 3600
        if delta_time_hrs <= 0:
            return 0

        return delta_load / delta_time_hrs

class CoincidentPeakPredictionSensor(SensorEntity):
    """
    Reworked Coincident Peak Prediction sensor that:
      - Always exposes the current instantaneous load as its state.
      - Tracks rolling load data to compute observed rate-of-change (ROC) and acceleration (ACC).
      - Uses a kinematic (quadratic) model to predict the daily peak time and load.
      - Switches between daily and short forecasts based on how close we are to the predicted peak.
      - Flags high-risk days based on 5CP logic.
    """
    ACCELERATION_THRESHOLD = 500  # MW/hr², easy to adjust centrally
    MAX_VALID_PEAK_WINDOW = 3    # hours
    SMOOTHING_ALPHA = 0.3  # between 0 (more smoothing) and 1 (less smoothing)

    def __init__(self, pjm_data, zone, peak_threshold, accuracy_threshold, sensor_type, hass):
        super().__init__()
        self.hass = hass
        self._pjm_data = pjm_data
        self._zone = zone
        self._sensor_type = sensor_type
        self._attr_name = f"Coincident Peak Prediction ({zone})"
        self._attr_unique_id = f"pjm_{sensor_type}_{zone}"
        self._unit_of_measurement = "MW"
        
        # The main sensor state is the current instantaneous load.
        self._state = None
        
        # Rolling load history (timestamp, load) for derivative calculations (~1-2 hours)
        self._load_history = deque(maxlen=12)
        
        # Forecast update trackers
        self._last_daily_forecast_update = None
        self._last_short_forecast_update = None
        self._last_kinematics_update = None
        
        # Predicted peak (from daily and short forecast refinements)
        self._predicted_peak = None
        self._predicted_peak_time = None

        # Observed derivatives from load history
        self._observed_roc = 0.0   # MW/hr
        self._observed_acc = 0.0   # MW/hr²

        # Forecast Variables
        self._daily_forecast_peak = None
        self._daily_forecast_peak_time = None
        self._short_forecast_peak = None
        self._short_forecast_peak_time = None
        self._kinematic_peak = None
        self._kinematic_peak_time = None
        self._forecasted_roc = 0.0
        self._forecasted_acc = 0.0
        
        # Bias factors to improve prediction over time
        self._roc_bias = 0.0
        self._acc_bias = 0.0

        # Adaptive bias factors for fine-tuning predicted time and magnitude
        self._time_bias = 0.0         # in hours
        self._magnitude_bias = 0.0    # in MW

        # Histories for adaptive learning
        self._time_error_history = deque(maxlen=30)       # errors in predicted time (hrs)
        self._magnitude_error_history = deque(maxlen=30)  # errors in predicted load (MW)

        # Initialize persistent storage for peaks
        self._store = Store(hass, 1, f"coincident_peaks_{zone}.json")
        self._top_five_peaks = []

        # Load peaks from storage
        self.hass.async_create_task(self._async_load_peaks())

        # Flags and thresholds
        self._daily_peak_occurred = False
        self._peak_threshold = peak_threshold
        self._accuracy_threshold = accuracy_threshold
        self._high_risk_day = False
        self._peak_hour_active = False
        self._error_history = deque(maxlen=30)

        # **Daily Reset**: Store the current date for which the prediction applies.
        self._current_prediction_date = dt_util.now().date()

        # Enable long-term statistics for zone load and system load
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def icon(self):
        return "mdi:summit"

    @property
    def unit_of_measurement(self):
        return self._unit_of_measurement

    @property
    def native_value(self):
        """Return the instantaneous load (MW) as the sensor state."""
        return self._state

    @property
    def extra_state_attributes(self):
        """Return additional predictive and diagnostic attributes."""
        formatted_top_five_peaks = [
            f"{timestamp.strftime('%B %d, %Y at %-I:%M:%S %p')} - {load:,.0f}"
            for timestamp, load in self._top_five_peaks
        ]
        
        return {
            "predicted_peak": self._predicted_peak,
            "predicted_peak_time": (
                self._predicted_peak_time.isoformat() 
                if self._predicted_peak_time else None
            ),
            "peak_hour_active": self._peak_hour_active,
            "high_risk_day": self._high_risk_day,
            "observed_roc": round(self._observed_roc, 2),
            "observed_acc": round(self._observed_acc, 2),
            "forecasted_roc": round(self._forecasted_roc, 2),
            "forecasted_acc": round(self._forecasted_acc, 2),
            "bias_roc": round(self._roc_bias, 2),
            "bias_acc": round(self._acc_bias, 2),
            "time_bias": round(self._time_bias, 2),
            "magnitude_bias": round(self._magnitude_bias, 2),
            "error_history": [round(err, 2) for err in self._error_history],
            "top_five_peaks": "\n".join(formatted_top_five_peaks),
        }

    async def async_update(self):
        """
        Main update flow executed on each sensor poll (e.g., every 5 minutes):
          1. Reset daily prediction if a new day has begun.
          2. Update instantaneous load and rolling history.
          3. Compute observed derivatives.
          4. If the daily peak hasn't occurred, update forecasts and refine predictions.
          5. Evaluate high-risk day and peak hour active status.
          6. After the predicted peak time, record forecast error and update adaptive biases.
        """
        now = dt_util.now()  # using Home Assistant's dt_util for timezone-aware times
        
        # *Daily Reset*: If a new day has started, reset the daily prediction.
        if now.date() != self._current_prediction_date:
            self._daily_peak_occurred = False
            self._predicted_peak = None
            self._predicted_peak_time = None
            self._last_daily_forecast_update = None
            self._last_short_forecast_update = None
            self._max_daily_load = None
            self._max_daily_load_time = None
            self._error_recorded = False
            self._current_prediction_date = now.date()

            self._daily_forecast_peak = None
            self._daily_forecast_peak_time = None
            self._short_forecast_peak = None
            self._short_forecast_peak_time = None
            self._kinematic_peak = None
            self._kinematic_peak_time = None
            self._forecasted_roc = 0.0
            self._forecasted_acc = 0.0

            _LOGGER.info("New day detected. Resetting daily peak predictions.")

        # Reset peaks if it's past October 1st and last reset was before October
        if now.date() >= date(now.year, 10, 1) and self._last_reset_date < date(now.year, 10, 1):
            self._top_five_peaks = []
            self._last_reset_date = date(now.year, 10, 1)
            await self._async_save_peaks()
            _LOGGER.info("Resetting peak history for new year (Oct 1st).")

        # 1. Update instantaneous load and record history.
        if not hasattr(self, '_last_load_update') or (now - self._last_load_update) >= timedelta(minutes=5):    
            success = await self._update_instantaneous_load()
            if success:
                now = dt_util.now()
                self._last_load_update = now
                # 2. Compute observed ROC and acceleration from load history.
                self._compute_observed_derivatives()
        
        # 3. Refine predictions if the daily peak has not occurred.
        await self._maybe_update_forecasts(now)

        # 4. Run Kinematics if within 1.5 hours of Predicted Peak, Short Forecast and load recently updated
        if not self._daily_peak_occurred:
            time_to_peak = (self._predicted_peak_time - now) if self._predicted_peak_time else None
            recently_updated = (self._last_short_forecast_update and (now - self._last_short_forecast_update < timedelta(minutes=10))) and \
                            (self._last_load_update and (now - self._last_load_update < timedelta(minutes=10)))

            if (time_to_peak and time_to_peak <= timedelta(minutes=110) and recently_updated):
                if not self._last_kinematics_update or (now - self._last_kinematics_update >= timedelta(minutes=5)):
                    self._predict_peak_using_kinematics(now)
                    self._last_kinematics_update = now
                    self._weighted_peak_prediction()
        
        # Check real-time load exceedance
        if self._state and self._predicted_peak and self._state > self._predicted_peak:
            self._predicted_peak = self._state
            self._predicted_peak_time = now
            _LOGGER.warning("Immediate peak adjustment due to real-time exceedance.")

        # 4. Evaluate high-risk day and peak hour active status.
        self._evaluate_5cp_risk()
        self._check_peak_hour_active(now)
        
        # 5. Once the predicted peak is past, record peak and forecast error.
        if (self._state is not None and self._predicted_peak is not None and
            self._observed_roc < 0 and
            self._max_daily_load is not None and
            self._max_daily_load >= 0.85 * self._predicted_peak and
            dt_util.now() >= self._predicted_peak_time and
            not getattr(self, "_error_recorded", False)):
            self._daily_peak_occurred = True
            self._record_error_and_update_bias()
            self._record_daily_peak()
            self._error_recorded = True
            _LOGGER.info("Peak detected: Actual peak %.1f MW at %s. Freezing further forecasts.",
                        self._max_daily_load, self._max_daily_load_time)

    async def _update_instantaneous_load(self):
        """Fetch the current load from PJMData and update state, load history, and maximum daily load."""
        try:
            load_val = await self._pjm_data.async_update_instantaneous(self._zone)
            if load_val is not None:
                now = dt_util.now()
                self._state = load_val
                self._load_history.append((now, load_val))
                
                #if self._state and self._predicted_peak and self._state > self._predicted_peak:
                #    _LOGGER.warning("Real-time load %.0f MW exceeds predicted peak %.0f MW. Adjusting immediately.", self._state, self._predicted_peak)
                #    self._predicted_peak = self._state
                #    self._predicted_peak_time = dt_util.now()

                # Update maximum daily load within this method.
                if not hasattr(self, "_max_daily_load") or self._max_daily_load is None:
                    self._max_daily_load = load_val
                    self._max_daily_load_time = now
                elif load_val > self._max_daily_load:
                    self._max_daily_load = load_val
                    self._max_daily_load_time = now
                return True
        except Exception as err:
            _LOGGER.error("Error updating instantaneous load: %s", err)
        return False # Indicate Failure

    def _compute_observed_derivatives(self):
        """
        Compute observed ROC (MW/hr) and ACC (MW/hr²) using a simple moving average (SMA)
        weighted by time, matching Home Assistant's derivative sensor algorithm.
        """

        if len(self._load_history) < 2:
            self._observed_roc = 0.0
            self._observed_acc = 0.0
            return

        sorted_history = sorted(self._load_history, key=lambda x: x[0])

        total_time_sec = 0.0
        weighted_roc_sum = 0.0

        for i in range(len(sorted_history) - 1):
            t0, val0 = sorted_history[i]
            t1, val1 = sorted_history[i + 1]
            delta_time_sec = (t1 - t0).total_seconds()
            if delta_time_sec <= 0:
                continue

            delta_load = val1 - val0
            interval_roc = delta_load / (delta_time_sec / 3600.0)  # MW/hr

            weighted_roc_sum += interval_roc * delta_time_sec
            total_time_sec += delta_time_sec

        if total_time_sec > 0:
            new_roc = weighted_roc_sum / total_time_sec
        else:
            new_roc = 0.0

        # Apply smoothing
        alpha = self.SMOOTHING_ALPHA
        self._observed_roc = (alpha * new_roc) + ((1 - alpha) * self._observed_roc)

        # Store ROC in history for ACC calculation
        if not hasattr(self, '_roc_history'):
            self._roc_history = deque(maxlen=12)
        self._roc_history.append((sorted_history[-1][0], self._observed_roc))

        # Compute acceleration (ACC)
        if len(self._roc_history) < 2:
            self._observed_acc = 0.0
            return

        sorted_roc_history = sorted(self._roc_history, key=lambda x: x[0])

        total_time_acc_sec = 0.0
        weighted_acc_sum = 0.0

        for i in range(len(sorted_roc_history) - 1):
            rt0, roc0 = sorted_roc_history[i]
            rt1, roc1 = sorted_roc_history[i + 1]
            delta_time_sec = (rt1 - rt0).total_seconds()
            if delta_time_sec <= 0:
                continue

            delta_roc = roc1 - roc0
            interval_acc = delta_roc / (delta_time_sec / 3600.0)

            weighted_acc_sum += interval_acc * delta_time_sec
            total_time_acc_sec += delta_time_sec

        if total_time_acc_sec > 0:
            new_acc = weighted_acc_sum / total_time_acc_sec
        else:
            new_acc = 0.0

        # Apply smoothing
        self._observed_acc = (alpha * new_acc) + ((1 - alpha) * self._observed_acc)

    async def _maybe_update_forecasts(self, now):
        """
        Decide whether to pull a daily forecast (if peak is far away) or a short forecast (within 3 hours).
        """
        if self._predicted_peak_time is None:
            _LOGGER.info("Predicted peak time is None. Fetching daily forecast (initialization).")
            await self._update_daily_forecast()
            self._last_daily_forecast_update = now
            self._weighted_peak_prediction()
            # After fetching, re-check predicted_peak_time
            if self._daily_forecast_peak_time:
                if now > self._daily_forecast_peak_time:
                    self._daily_peak_occurred = True
                    _LOGGER.info("Initialization: Peak has already passed for today. No further forecasts.")
                    return  # Peak already passed, no further action needed
            else:
                _LOGGER.warning("Daily forecast peak time still None after initialization.")
                return

        # Regular operational check after initialization
        forecast_peak_time = self._predicted_peak_time #or self._daily_forecast_peak_time

        if forecast_peak_time and now > forecast_peak_time:
            self._daily_peak_occurred = True
            #_LOGGER.info("Predicted peak occurred -  now > forecast_peak_time, Set Daily Peak Occurred")
            return

        time_to_peak = forecast_peak_time - now if forecast_peak_time else None
        _LOGGER.info("Maybe Update Forecast - Time to peak:", time_to_peak)

        if (time_to_peak is None or time_to_peak > timedelta(hours=2)) and (
            not self._last_daily_forecast_update or (now - self._last_daily_forecast_update) >= timedelta(hours=1)
        ):
            await self._update_daily_forecast()
            self._last_daily_forecast_update = now
            self._weighted_peak_prediction()
        elif time_to_peak <= timedelta(hours=3) and not self._daily_peak_occurred:
            if (not self._last_short_forecast_update or (now - self._last_short_forecast_update) >= timedelta(minutes=5)):
                await self._update_short_forecast()
                self._last_short_forecast_update = now
                self._weighted_peak_prediction()

    async def _update_daily_forecast(self):
        """Pull daily forecast data and update predicted peak and time for today."""
        try:
            forecast_zone = "RTO_COMBINED" if self._zone.upper() == "PJM RTO" else self._zone
            data = await self._pjm_data.async_update_forecast(forecast_zone)
            if data:
                today = dt_util.now().date()
                day_data = [x for x in data if x["forecast_hour_ending"].date() == today]
                if day_data:
                    max_item = max(day_data, key=lambda x: x["forecast_load_mw"])
                    self._daily_forecast_peak = max_item["forecast_load_mw"]
                    self._daily_forecast_peak_time = max_item["forecast_hour_ending"] - timedelta(hours=1)
                    _LOGGER.info("Daily forecast: peak=%.1f at %s", self._daily_forecast_peak, self._daily_forecast_peak_time)
                    #self._predicted_peak = max_item["forecast_load_mw"]
                    #self._predicted_peak_time = max_item["forecast_hour_ending"]
        except Exception as err:
            _LOGGER.error("Error updating daily forecast: %s", err)

    async def _update_short_forecast(self):
        """
        Pull short forecast data to compute forecasted derivatives (if available) and update
        separate short-term forecast attributes. These attributes are then used in weighted predictions.
        """
        try:
            forecast_zone = "RTO_COMBINED" if self._zone.upper() == "PJM RTO" else self._zone
            data = await self._pjm_data.async_update_short_forecast(forecast_zone)
            if data and len(data) > 1:
                # Calculate forecasted derivatives for kinematic prediction
                times, loads = self._extract_time_load_arrays_short(data, limit_minutes=60)
                if len(times) >= 3:
                    coeffs = np.polyfit(times, loads, 2)
                    t_last = times[-1]
                    self._forecasted_roc = 2 * coeffs[0] * t_last + coeffs[1]
                    self._forecasted_acc = 2 * coeffs[0]
                else:
                    self._forecasted_roc = 0.0
                    self._forecasted_acc = 0.0

                # Separately store short forecast peak for weighted prediction
                max_item = max(data, key=lambda x: x["forecast_load_mw"])
                if max_item != data[-1]:
                    self._short_forecast_peak = max_item["forecast_load_mw"]
                    self._short_forecast_peak_time = max_item["forecast_hour_ending"]
                else:
                    self._short_forecast_peak = None
                    self._short_forecast_peak_time = None
            else:
                # Clear if insufficient data
                self._forecasted_roc = 0.0
                self._forecasted_acc = 0.0
                self._short_forecast_peak = None
                self._short_forecast_peak_time = None

        except Exception as err:
            _LOGGER.error("Error updating short forecast: %s", err)
            self._forecasted_roc = 0.0
            self._forecasted_acc = 0.0
            self._short_forecast_peak = None
            self._short_forecast_peak_time = None

    def _predict_peak_using_kinematics(self, now):
        """
        Use the current load, observed ROC and ACC (optionally blended with forecasted values
        and bias adjustments) to predict the peak time and load.
        Applies adaptive blending based on time_to_peak.
        """
        # Thresholds and constants
        ACCELERATION_THRESHOLD = 50  # Min abs acceleration magnitude (MW/hr²) - TUNE THIS
        MIN_DECELERATION = 50       # Min abs negative acceleration (MW/hr²) - TUNE THIS
        MAX_VALID_PEAK_WINDOW_HRS = 4 # Max hours ahead for valid prediction

        _LOGGER.info( # Log raw inputs once
            "Kinematic inputs: current_load=%.1f, obs_roc=%.2f, obs_acc=%.2f, "
            "fcst_roc=%.2f, fcst_acc=%.2f, roc_bias=%.2f, acc_bias=%.2f",
            self._state or 0.0, self._observed_roc, self._observed_acc,
            self._forecasted_roc, self._forecasted_acc, self._roc_bias, self._acc_bias
        )

        # Check if forecast derivatives are available
        if self._forecasted_roc == 0 and self._forecasted_acc == 0: # Allow if only one is zero maybe?
            _LOGGER.info("Skipping kinematic: Forecasted ROC/ACC insufficient.")
            self._kinematic_peak = None; self._kinematic_peak_time = None
            return

        # --- Adaptive Blending ---
        time_to_peak_hrs = float('inf') # Default to infinity if no prediction yet
        if self._predicted_peak_time:
            time_diff_sec = (self._predicted_peak_time - now).total_seconds()
            if time_diff_sec > 0: # Only consider future predicted peaks
                time_to_peak_hrs = time_diff_sec / 3600.0

        # Define weighting function (linear ramp example)
        # Weight shifts towards observed data as predicted time approaches.
        # obs_weight = 1.0 means fully observed, 0.0 means fully forecast.
        # Range: From 0.2 obs weight at 2+ hrs away, up to 0.8 obs weight at 0.5 hrs or less.
        obs_weight = np.clip(1.0 - (time_to_peak_hrs - 0.5) / (2.0 - 0.5), 0.1, 0.9)
        fcst_weight = 1.0 - obs_weight
        #_LOGGER.info(f"Adaptive Blend Weights (time_to_peak={time_to_peak_hrs:.2f} hrs): obs={obs_weight:.2f}, fcst={fcst_weight:.2f}")

        blended_roc = (obs_weight * self._observed_roc + fcst_weight * self._forecasted_roc) + self._roc_bias
        blended_acc = (obs_weight * self._observed_acc + fcst_weight * self._forecasted_acc) + self._acc_bias
        # --- End Adaptive Blending ---


        #_LOGGER.info(
        #    "Kinematic blended values: blended_roc=%.2f, blended_acc=%.2f",
        #    blended_roc, blended_acc
        #)

        # --- Guard Conditions ---
        if blended_acc == 0:
        #    _LOGGER.info("Skipping kinematic: blended_acc is zero.")
            self._kinematic_peak = None; self._kinematic_peak_time = None
            return
        if blended_acc >= 0: # Require deceleration
        #    _LOGGER.info("Skipping kinematic: blended_acc (%.2f) is non-negative.", blended_acc)
            self._kinematic_peak = None; self._kinematic_peak_time = None
            return
        if abs(blended_acc) < MIN_DECELERATION: # Require minimum deceleration magnitude
        #    _LOGGER.info("Skipping kinematic: blended_acc magnitude |%.2f| < threshold %.1f.", blended_acc, MIN_DECELERATION)
            self._kinematic_peak = None; self._kinematic_peak_time = None
            return
        # --- End Guards ---

        # Calculate raw t_peak
        raw_t_peak = -blended_roc / blended_acc
        #_LOGGER.info("Kinematic raw calculation: raw_t_peak=%.2f hrs", raw_t_peak)

        # Apply adaptive time bias
        t_peak = raw_t_peak + self._time_bias
        #_LOGGER.info("Kinematic after time bias: t_peak=%.2f hrs", t_peak)

        # Check final time validity
        if not (0 < t_peak <= MAX_VALID_PEAK_WINDOW_HRS):
        #    _LOGGER.info(
        #        "Predicted peak time %.2f hrs (adjusted) invalid or unrealistic (raw: %.2f hrs). Ignoring.",
        #        t_peak, raw_t_peak
        #    )
            self._kinematic_peak = None; self._kinematic_peak_time = None
            return

        # Calculate predicted load using blended values and adjusted time
        if self._state is None: # Need current load for prediction
            _LOGGER.warning("Skipping kinematic load calculation: current state is None.")
            self._kinematic_peak = None; self._kinematic_peak_time = None
            return
            
        predicted_load = self._state + blended_roc * t_peak + 0.5 * blended_acc * (t_peak ** 2)
        predicted_load += self._magnitude_bias

        # Update kinematic state
        self._kinematic_peak = int(round(predicted_load))
        self._kinematic_peak_time = now + timedelta(hours=t_peak)

        _LOGGER.info(
            "Kinematic Prediction SUCCESS: Peak=%.0f MW at %s (in %.2f hrs)",
            self._kinematic_peak, self._kinematic_peak_time.isoformat(), t_peak
        )

    def _weighted_peak_prediction(self):
        now = dt_util.now()
        predictions = []
        weights = []

        # Kinematic prediction (most sensitive)
        if self._kinematic_peak_time and self._kinematic_peak:
            time_diff = abs((self._kinematic_peak_time - now).total_seconds() / 3600)
            if time_diff < 3:
                predictions.append((self._kinematic_peak_time, self._kinematic_peak))
                weights.append(0.5)  # high weight near peak time

        # Short-term forecast
        if self._short_forecast_peak_time and self._short_forecast_peak:
            predictions.append((self._short_forecast_peak_time, self._short_forecast_peak))
            weights.append(0.3)

        # Daily forecast (less sensitive, fallback)
        if self._daily_forecast_peak_time and self._daily_forecast_peak:
            predictions.append((self._daily_forecast_peak_time, self._daily_forecast_peak))
            weights.append(0.2)
            _LOGGER.info("Daily forecast appended:", self._daily_forecast_peak, self._daily_forecast_peak_time)
        if not predictions:
            _LOGGER.info("No valid predictions available after weighted peak prediction calculation.")
            return  # no valid predictions yet

        # Weighted averaging
        peak_time = sum((p[0].timestamp() * w for p, w in zip(predictions, weights))) / sum(weights)
        peak_magnitude = sum((p[1] * w for p, w in zip(predictions, weights))) / sum(weights)
        #_LOGGER.info("Weighted Average:", peak_magnitude, peak_time)
        self._predicted_peak_time = datetime.fromtimestamp(peak_time, tz=timezone.utc)
        self._predicted_peak = peak_magnitude

    def _evaluate_5cp_risk(self):
        """Flag high-risk day if predicted peak is near or exceeds the 5th highest historical peak."""
        if not self._predicted_peak:
            self._high_risk_day = False
            return
        fifth_peak = self._get_fifth_highest_peak()
        self._high_risk_day = self._predicted_peak >= 0.95 * fifth_peak

    def _check_peak_hour_active(self, now):
        """
        Set peak_hour_active True if the current time falls within the hour of the predicted peak,
        and if the day is flagged as high-risk.
        """
        if not self._high_risk_day or not self._predicted_peak_time:
            self._peak_hour_active = False
            return
        pstart = self._predicted_peak_time.replace(minute=0, second=0, microsecond=0)
        pend = pstart + timedelta(hours=1)
        self._peak_hour_active = (pstart <= now < pend)

    def _record_error_and_update_bias(self):
        """
        After the peak has passed, compare the actual peak load (from recent history)
        with the predicted peak load. Record the error and adjust bias factors accordingly.
        """
        # Initial check: Make sure the necessary source attributes are present
        if not self._max_daily_load or not self._predicted_peak or not self._predicted_peak_time:
            # Changed level to DEBUG as this is normal operation if conditions not met
            _LOGGER.debug("Skipping error recording: Missing max_daily_load, predicted_peak, or predicted_peak_time.")
            return

        # Assign local variables AFTER the initial check ---
        actual_peak = self._max_daily_load
        actual_peak_time = self._max_daily_load_time

        # --- Robust Checks Before Calculation ---
        if actual_peak is None or self._predicted_peak is None:
            _LOGGER.warning(
                "Cannot calculate magnitude error: actual_peak (%s) or predicted_peak (%s) is None.",
                actual_peak, self._predicted_peak
            )
            magnitude_error = 0 # Assign a default or skip magnitude bias update
        else:
            try:
                magnitude_error = actual_peak - self._predicted_peak
            except TypeError as e:
                _LOGGER.error(
                    "TypeError calculating magnitude_error (actual=%s, predicted=%s): %s",
                    actual_peak, self._predicted_peak, e
                )
                magnitude_error = 0 # Assign a default

        if actual_peak_time is None or self._predicted_peak_time is None:
            _LOGGER.warning(
                "Cannot calculate time error: actual_peak_time (%s) or predicted_peak_time (%s) is None.",
                actual_peak_time, self._predicted_peak_time
            )
            time_error = 0 # Assign a default or skip time bias update
        else:
            try:
                # Ensure both are datetime objects before subtraction
                if isinstance(actual_peak_time, datetime) and isinstance(self._predicted_peak_time, datetime):
                    time_error = (actual_peak_time - self._predicted_peak_time).total_seconds() / 3600
                else:
                    _LOGGER.error("TypeError calculating time_error: Operands are not both datetimes (actual=%s, predicted=%s)", type(actual_peak_time), type(self._predicted_peak_time))
                    time_error = 0 # Default on type error
            except TypeError as e: # Catch potential errors during subtraction itself
                _LOGGER.error(
                    "TypeError calculating time_error (actual=%s, predicted=%s): %s",
                    actual_peak_time, self._predicted_peak_time, e
                )
                time_error = 0 # Assign a default
        # --- End Robust Checks ---


        # Proceed with history update and bias calculation (using potentially defaulted errors)
        self._time_error_history.append(time_error)
        self._magnitude_error_history.append(magnitude_error)
        # Check if error_history should track time or magnitude error? Currently magnitude.
        self._error_history.append(magnitude_error)

        # Calculate average errors from history.
        avg_time_error = np.mean(list(self._time_error_history)) if self._time_error_history else 0
        avg_magnitude_error = np.mean(list(self._magnitude_error_history)) if self._magnitude_error_history else 0

        # Update adaptive bias factors.
        # Only update if the error calculation was likely valid (optional refinement)
        if time_error != 0 or len(self._time_error_history) == 1:
            self._time_bias -= 0.1 * avg_time_error
        if magnitude_error != 0 or len(self._magnitude_error_history) == 1:
            self._magnitude_bias -= 0.1 * avg_magnitude_error

        # Optional: Add bias clipping here if desired
        # MAX_TIME_BIAS_HOURS = 2.0
        # MAX_MAGNITUDE_BIAS_MW = 5000
        # self._time_bias = max(-MAX_TIME_BIAS_HOURS, min(MAX_TIME_BIAS_HOURS, self._time_bias))
        # self._magnitude_bias = max(-MAX_MAGNITUDE_BIAS_MW, min(MAX_MAGNITUDE_BIAS_MW, self._magnitude_bias))


        _LOGGER.info(
            "Peak occurred: Actual=%.1f MW @ %s, Predicted=%.1f MW @ %s, "
            "Time Error=%.2f hrs, Magnitude Error=%.1f MW, "
            "Updated adaptive biases: time_bias=%.2f, magnitude_bias=%.1f",
            actual_peak if actual_peak is not None else float('nan'), # Handle None for logging
            actual_peak_time,
            self._predicted_peak if self._predicted_peak is not None else float('nan'), # Handle None for logging
            self._predicted_peak_time,
            time_error, magnitude_error, self._time_bias, self._magnitude_bias
        )

    def _get_fifth_highest_peak(self):
        """
        Return the 5th highest peak from the known top peaks or the user-defined threshold if fewer than 5.
        """
        if len(self._top_five_peaks) < 5:
            return self._peak_threshold
        return sorted(self._top_five_peaks, key=lambda x: x[1], reverse=True)[4][1]


    def _record_daily_peak(self):
        """Record the day's peak load and maintain the top five unique daily peaks."""
        peak_date = self._max_daily_load_time.date()
        updated = False

        # Check if today's date is already recorded
        for i, (ts, val) in enumerate(self._top_five_peaks):
            if ts.date() == peak_date:
                if self._max_daily_load > val:
                    _LOGGER.debug("Updating today's peak from %.1f MW to %.1f MW", val, self._max_daily_load)
                    self._top_five_peaks[i] = (self._max_daily_load_time, self._max_daily_load)
                    updated = True
                else:
                    _LOGGER.debug("Today's peak (%s) already recorded and current peak %.1f MW is not higher than %.1f MW", peak_date, self._max_daily_load, val)
                break

        # If not already recorded, add the new peak
        if not updated and all(ts.date() != peak_date for ts, _ in self._top_five_peaks):
            _LOGGER.debug("Adding new daily peak for %s: %.1f MW", peak_date, self._max_daily_load)
            self._top_five_peaks.append((self._max_daily_load_time, self._max_daily_load))

        # Sort and trim the list to the top 5 peaks
        self._top_five_peaks.sort(key=lambda x: x[1], reverse=True)
        if len(self._top_five_peaks) > 5:
            removed_peak = self._top_five_peaks.pop()
            _LOGGER.debug("Removing lowest peak %s", removed_peak)

        # Save updated peaks
        self.hass.async_create_task(self._async_save_peaks())

    def _extract_time_load_arrays(self, history_deque, limit_hours=1.0):
        """
        Extract data from the rolling history for the past 'limit_hours' and convert times to hours
        since the earliest timestamp.
        """
        now = dt_util.now()
        earliest = now - timedelta(hours=limit_hours)
        filtered = [(ts, val) for (ts, val) in history_deque if ts >= earliest]
        if not filtered:
            return np.array([]), np.array([])
        filtered.sort(key=lambda x: x[0])
        base_time = filtered[0][0]
        times = [(ts - base_time).total_seconds() / 3600.0 for (ts, _) in filtered]
        loads = [val for (_, val) in filtered]
        return np.array(times), np.array(loads)

    def _extract_time_load_arrays_short(self, forecast_data, limit_minutes=60):
        """
        Convert the short forecast data (list of dicts) into time (in hours) and load arrays,
        limited to the first 'limit_minutes' of forecast.
        """
        base_time = forecast_data[0]["forecast_hour_ending"]
        cutoff = base_time + timedelta(minutes=limit_minutes)
        subset = [item for item in forecast_data if item["forecast_hour_ending"] <= cutoff]
        if not subset:
            return np.array([]), np.array([])
        subset.sort(key=lambda x: x["forecast_hour_ending"])
        times = [(item["forecast_hour_ending"] - base_time).total_seconds() / 3600.0 for item in subset]
        loads = [item["forecast_load_mw"] for item in subset]
        return np.array(times), np.array(loads)

    async def _async_load_peaks(self):
        data = await self._store.async_load()
        if data:
            raw_peaks = data.get('top_five_peaks', [])
            self._top_five_peaks = [
                (dt_util.parse_datetime(timestamp), load)
                for timestamp, load in raw_peaks
            ]
            self._last_reset_date = date.fromisoformat(data.get('last_reset_date'))
        else:
            self._top_five_peaks = []
            self._last_reset_date = date.today()

    async def _async_save_peaks(self):
        await self._store.async_save({
            'top_five_peaks': self._top_five_peaks,
            'last_reset_date': self._last_reset_date.isoformat(),
        })

class PJMData:
    """Get and parse data from PJM with coordinated API rate limiting using your API key or fetched subscription key."""
    def __init__(self, websession, api_key):
        self._websession = websession
        self._subscription_key = api_key
        self._request_times = deque(maxlen=6)
        self._lock = asyncio.Lock()

    async def _rate_limit(self):
        async with self._lock:
            now = time_module.time()
            # Remove timestamps older than 60 seconds
            while self._request_times and now - self._request_times[0] >= 60:
                self._request_times.popleft()

            if len(self._request_times) >= 6:
                wait_time = 60 - (now - self._request_times[0]) + 1
                _LOGGER.warning("PJM API rate limit reached. Waiting %.2f seconds.", wait_time)
                await asyncio.sleep(wait_time)
                # Clean up again after sleep
                now = time_module.time()
                while self._request_times and now - self._request_times[0] >= 60:
                    self._request_times.popleft()

            self._request_times.append(now)

    def _get_headers(self):
        return {
            'Ocp-Apim-Subscription-Key': self._subscription_key,
            'Content-Type': 'application/json',
        }

    async def _get_subscription_key(self):
        if self._subscription_key:
            return
        try:
            with async_timeout.timeout(60):
                response = await self._websession.get(RESOURCE_SUBSCRIPTION_KEY)
                data = await response.json()
                self._subscription_key = data.get('subscriptionKey')
                if not self._subscription_key:
                    _LOGGER.error("No subscription key found in response from %s", RESOURCE_SUBSCRIPTION_KEY)
        except Exception as err:
            _LOGGER.error("Failed to get subscription key: %s", err)

    async def async_update_instantaneous(self, zone):
        retries = 3
        backoff = 10  # seconds
        for attempt in range(retries):
            await self._rate_limit()
            if not self._subscription_key:
                await self._get_subscription_key()

            end_time_utc = datetime.now(timezone.utc)
            start_time_utc = end_time_utc - timedelta(minutes=10)
            time_string = start_time_utc.strftime('%m/%e/%Y %H:%Mto') + end_time_utc.strftime('%m/%e/%Y %H:%M')
            params = {
                'rowCount': '100',
                'sort': 'datetime_beginning_utc',
                'order': 'Desc',
                'startRow': '1',
                'isActiveMetadata': 'true',
                'fields': 'area,instantaneous_load',
                'datetime_beginning_utc': time_string,
            }
            resource = f"{RESOURCE_INSTANTANEOUS}?{urllib.parse.urlencode(params)}"
            headers = self._get_headers()

            try:
                with async_timeout.timeout(60):
                    response = await self._websession.get(resource, headers=headers)
                    if response.status == 429:
                        _LOGGER.warning("PJM API rate limit exceeded (429). Retrying in %d seconds.", backoff)
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    data = await response.json()
                    if not data:
                        _LOGGER.error("No load data returned for zone %s", zone)
                        return None

                    items = data["items"]
                    for item in items:
                        if item["area"] == zone:
                            return int(round(item["instantaneous_load"]))

                    _LOGGER.error("Couldn't find load data for zone %s", zone)
                    return None

            except (asyncio.TimeoutError, aiohttp.ClientError) as err:
                _LOGGER.error("Could not get load data from PJM: %s", err)
                await asyncio.sleep(backoff)
                backoff *= 2
            except Exception as err:
                _LOGGER.error("Unexpected error fetching load data: %s", err)
                return None

        _LOGGER.error("Exhausted retries to get instantaneous load data.")
        return None

    async def async_update_forecast(self, zone):
        retries = 3
        backoff = 10
        for attempt in range(retries):
            await self._rate_limit()
            if not self._subscription_key:
                await self._get_subscription_key()

            midnight_local = datetime.combine(date.today(), time())
            start_time_utc = midnight_local.astimezone(timezone.utc)
            end_time_utc = start_time_utc + timedelta(hours=23, minutes=59)
            time_string = start_time_utc.strftime('%m/%e/%Y %H:%Mto') + end_time_utc.strftime('%m/%e/%Y %H:%M')
            params = {
                'rowCount': '100',
                'order': 'Asc',
                'startRow': '1',
                'isActiveMetadata': 'true',
                'fields': 'forecast_datetime_ending_utc,forecast_load_mw',
                'forecast_datetime_beginning_utc': time_string,
                'forecast_area': zone,
            }
            resource = f"{RESOURCE_FORECAST}?{urllib.parse.urlencode(params)}"
            headers = self._get_headers()

            try:
                with async_timeout.timeout(60):
                    response = await self._websession.get(resource, headers=headers)
                    if response.status == 429:
                        _LOGGER.warning("PJM API rate limit exceeded (429). Retrying in %d seconds.", backoff)
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    full_data = await response.json()
                    data = full_data["items"]
                    forecast_data = []
                    for item in data:
                        forecast_hour_ending = datetime.strptime(item['forecast_datetime_ending_utc'], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone()
                        forecast_data.append({
                            "forecast_hour_ending": forecast_hour_ending,
                            "forecast_load_mw": int(item["forecast_load_mw"])
                        })
                    return forecast_data

            except (asyncio.TimeoutError, aiohttp.ClientError) as err:
                _LOGGER.error("Could not get forecast data from PJM: %s", err)
                await asyncio.sleep(backoff)
                backoff *= 2
            except Exception as err:
                _LOGGER.error("Unexpected error fetching forecast data: %s", err)
                return None

        _LOGGER.error("Exhausted retries to get forecast data.")
        return None

    async def async_update_short_forecast(self, zone):
        retries = 3
        backoff = 5  # seconds
        for attempt in range(retries):
            await self._rate_limit()
            if not self._subscription_key:
                await self._get_subscription_key()

            params = {
                'rowCount': '48',
                'order': 'Asc',
                'startRow': '1',
                'fields': 'forecast_datetime_ending_utc,forecast_load_mw',
                'evaluated_at_ept': '5MinutesAgo',
                'forecast_area': zone,
            }
            resource = f"{RESOURCE_SHORT_FORECAST}?{urllib.parse.urlencode(params)}"
            headers = self._get_headers()

            try:
                with async_timeout.timeout(60):
                    response = await self._websession.get(resource, headers=headers)
                    if response.status == 429:
                        _LOGGER.warning("PJM API rate limit exceeded (429). Retrying in %d seconds.", backoff)
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    full_data = await response.json()
                    data = full_data["items"]

                    forecast_data = []
                    for item in data:
                        forecast_hour_ending = datetime.strptime(item['forecast_datetime_ending_utc'], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone()
                        forecast_data.append({
                            "forecast_hour_ending": forecast_hour_ending,
                            "forecast_load_mw": int(item["forecast_load_mw"])
                        })
                    return forecast_data

            except (asyncio.TimeoutError, aiohttp.ClientError) as err:
                _LOGGER.error("Could not get short forecast data from PJM: %s", err)
                await asyncio.sleep(backoff)
                backoff *= 2
            except Exception as err:
                _LOGGER.error("Unexpected error fetching short forecast data: %s", err)
                return None

        _LOGGER.error("Exhausted retries to get short forecast data.")
        return None

    async def async_update_lmp(self, pnode_id):
        retries = 3
        backoff = 10  # initial backoff interval in seconds
        for attempt in range(retries):
            await self._rate_limit()  # Ensure we respect the overall API rate limit

            # Fetch subscription key if it's not already available
            if not self._subscription_key:
                await self._get_subscription_key()

            # Define the time range for the LMP query (past ~1 hour)
            now_utc = datetime.now(timezone.utc)
            current_minute = now_utc.minute

            if current_minute < 5:
                # Adjust start time to the previous hour if current minute < 5
                start_time_utc = now_utc.replace(minute=4, second=0, microsecond=0) - timedelta(hours=1)
            else:
                start_time_utc = now_utc.replace(minute=4, second=0, microsecond=0)

            time_string = start_time_utc.strftime('%m/%e/%Y %H:%Mto') + now_utc.strftime('%m/%e/%Y %H:%M')

            params = {
                'rowCount': '12',
                'order': 'Asc',
                'startRow': '1',
                'datetime_beginning_utc': time_string,
                'pnode_id': pnode_id,
            }

            resource = f"{RESOURCE_LMP}?{urllib.parse.urlencode(params)}"
            headers = self._get_headers()

            try:
                with async_timeout.timeout(60):
                    response = await self._websession.get(resource, headers=headers)

                    if response.status == 429:
                        # API rate limit exceeded; perform exponential backoff and retry
                        _LOGGER.warning("PJM API rate limit exceeded (429) while fetching LMP data. Retrying in %d seconds.", backoff)
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    data = await response.json()
                    if not data:
                        _LOGGER.error("No LMP data returned for pnode_id %s", pnode_id)
                        return None

                    items = data["items"]

                    # Extract total LMP values specific to the requested pnode_id
                    total_lmp_values = [float(item["total_lmp_rt"]) for item in items if item["pnode_id"] == pnode_id]

                    if not total_lmp_values:
                        _LOGGER.error("Couldn't find LMP data for pnode_id %s", pnode_id)
                        return None

                    # Calculate and return the average LMP value
                    average_lmp = sum(total_lmp_values) / len(total_lmp_values)
                    return round(average_lmp, 2)

            except (asyncio.TimeoutError, aiohttp.ClientError) as err:
                # Handle network-related errors and apply exponential backoff
                _LOGGER.error("Could not get LMP avg data from PJM: %s", err)
                await asyncio.sleep(backoff)
                backoff *= 2
            except Exception as err:
                # Catch-all for other exceptions
                _LOGGER.error("Unexpected error fetching LMP avg data: %s", err)
                return None

        # All retries have been exhausted
        _LOGGER.error("Exhausted retries to get LMP data.")
        return None
