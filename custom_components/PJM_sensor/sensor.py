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
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import Throttle
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
                peak_threshold, accuracy_threshold, sensor_type))
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
            attr["forecast_hour_ending"] = self._forecast_hour_ending.isoformat() if hasattr(self, "_forecast_hour_ending") and self._forecast_hour_ending else None
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
    Enhanced sensor to predict PJM (or zone) coincident peaks using a multi-stage approach:

      1. Daily 5CP Likelihood by 03:00
         - Compare today's 7-day forecast peak to the current 5th highest system peak.
         - If forecast is near or above that threshold, mark a 'high CP risk' day.
      
      2. Intraday Hour-Ahead Prediction
         - Continuously refine the exact peak hour using:
             • Five-Minute Load Forecast (short forecast),
             • Instantaneous Load,
             • Real-time temperature (optional, if exposed in the API),
             • Observed forecast errors.
         - Provide at least 1 hour lead time if possible.
      
      3. Self-Improving Over Time
         - Each day, track whether a peak event was correctly identified.
         - Update top 5 peaks list and compare day’s actual peak with forecasted.
         - Adjust daily threshold or forecast bias if consistent over/underpredictions are observed.
    """

    def __init__(self, pjm_data, zone, peak_threshold, accuracy_threshold, sensor_type):
        super().__init__()
        self._pjm_data = pjm_data
        self._zone = zone
        self._sensor_type = sensor_type
        self._attr_name = f"Coincident Peak Prediction ({zone})"
        self._attr_unique_id = f"pjm_{sensor_type}_{zone}"
        self._unit_of_measurement = "MW"
        self._state = None

        # --- Track last API update
        self._last_load_update = None
        self._last_forecast_update = None
        self._last_short_forecast_update = None

        # --- Configurable thresholds
        self._user_defined_threshold = peak_threshold
        self._accuracy_threshold = accuracy_threshold

        # --- Data Structures
        self._load_history = deque(maxlen=MAX_HISTORY_SIZE) 
        self._daily_forecast = []
        self._actual_daily_peaks = {}         
        self._forecasted_daily_peaks = {}     
        self._forecasted_daily_peak_time = {} 
        self._top_five_peaks = []
        self._short_forecast_history = []
        self._historical_peak_accuracy = []

        # --- Additional attributes for daily CP warnings
        self._forecasted_peak_today = False
        self._peak_hour_active = False
        self._predicted_peak_time = None
        self._refined_forecast = None

        # --- New: Track days flagged as potential CP (so we can refine intraday)
        self._high_risk_day = False
        self._morning_prediction_time = None

        # For storing typical forecast bias or error metrics (to self-improve).
        self._rolling_forecast_errors = deque(maxlen=30)  # store last 30 day-peak errors

    @property
    def icon(self):
        return "mdi:summit"

    @property
    def unit_of_measurement(self):
        return self._unit_of_measurement

    @property
    def native_value(self):
        """
        For immediate reference: the predicted peak load (MW).
        """
        return self._state

    @property
    def extra_state_attributes(self):
        """
        Returns a dict with additional debug / tracking fields.
        """
        return {
            "predicted_peak": self._state,
            "predicted_peak_time": self._predicted_peak_time.isoformat() if self._predicted_peak_time else None,
            "forecasted_peak_today": self._forecasted_peak_today,
            "peak_hour_active": self._peak_hour_active,
            "high_risk_day": self._high_risk_day,
            "top_five_peaks": self._top_five_peaks,
            "recent_forecast_errors": list(self._rolling_forecast_errors),
            "accuracy_probability_percent": self._compute_accuracy_probability(),
        }

    async def async_update(self):
        """
        Main update flow. 
          1) Fetch real-time load + short-term forecast,
          2) Possibly do the daily 03:00 CP check,
          3) If high risk, refine intraday peak predictions,
          4) Update the state & flags accordingly,
          5) Track performance.
        """
        now_local = datetime.now(timezone.utc).astimezone()
        load = self._state

        # 1. Get latest instantaneous load.
        if self._last_load_update is None or (now_local - self._last_load_update) >= timedelta(minutes=5):
            load = await self._pjm_data.async_update_instantaneous(self._zone)
            if load is not None:
                self._append_load_history(load)
                self._last_load_update = now_local  # Update last load fetch timestamp

        # 2. Each Hour perform a 5CP check. 
        if (
            self._morning_prediction_time is None or 
            self._morning_prediction_time.date() < now_local.date() or 
            (now_local - self._morning_prediction_time) >= timedelta(hours=1)
        ):
            self._morning_prediction_time = now_local
            await self._daily_5cp_check(now_local)

        # 3. Update actual daily peaks from the load history.
        self._update_actual_daily_peaks(now_local)

        # 4. If the day was flagged high risk, refine predictions intraday:
        if self._high_risk_day:
            now_local = datetime.now(timezone.utc).astimezone()

            # Refresh short forecast only within 3 hours of the daily forecast peak hour
            forecast_peak_hour = self._forecasted_daily_peak_time.get(now_local.date())
            if forecast_peak_hour and abs((forecast_peak_hour - now_local).total_seconds()) <= 3 * 3600:
                if (self._last_short_forecast_update is None or 
                    (now_local - self._last_short_forecast_update) >= timedelta(minutes=5)):
                    short_forecast_data = await self._pjm_data.async_update_short_forecast(self._zone)
                    if short_forecast_data:
                        max_short_forecast = max(short_forecast_data, key=lambda x: x["forecast_load_mw"])
                        short_load = max_short_forecast["forecast_load_mw"]
                        short_time = max_short_forecast["forecast_hour_ending"]
                        self._last_short_forecast_update = now_local
                    else:
                        short_load = None
                        short_time = None

                # Fallback to daily forecast if short forecast unavailable
                if short_load is None:
                    short_load = self._forecasted_daily_peaks.get(now_local.date(), load)
                    short_time = forecast_peak_hour

                # Calculate forecasted rate of change (first derivative)
                forecasted_rate = self._calculate_forecasted_rate_of_change(short_forecast_data)

                # Calculate instantaneous load rate of change
                instantaneous_rate = self._calculate_instantaneous_rate_of_change()

                # Determine if actual peak is arriving sooner/later
                time_adjustment = self._predict_time_shift(instantaneous_rate, forecasted_rate)

                # Adjust predicted peak time based on rate comparison
                refined_peak_time = short_time + timedelta(minutes=time_adjustment)

                # Update predictions
                self._state = short_load
                self._predicted_peak_time = refined_peak_time
                self._forecasted_peak_today = self._state >= self._get_fifth_highest_peak()
            else:
                # Too early or late for refinement; just use daily forecast
                self._state = self._forecasted_daily_peaks.get(now_local.date(), load)
                self._predicted_peak_time = forecast_peak_hour
                self._forecasted_peak_today = self._state >= self._get_fifth_highest_peak()
        else:
            # Not high-risk; default to instantaneous load
            self._state = load
            self._predicted_peak_time = None
            self._forecasted_peak_today = False

        # 5. Check if we are currently in the predicted peak hour:
        self._peak_hour_active = self._check_if_peak_hour_active(now_local)

        # 6. Update top five peaks.
        self._update_top_five_peaks()

        # 7. Log accuracy if near the end of the day or top-of-hour.
        if now_local.minute < 3:
            self._record_forecast_error(load)

    async def _daily_5cp_check(self, now_local):
        """
        Do the morning check to see if today's forecast might exceed the current 5th highest peak.
        If yes, set high_risk_day = True, so we do more detailed intraday tracking.
        """
        # Pull the 1-day forecast for the zone:
        forecast_data = await self._pjm_data.async_update_forecast(self._zone)
        if not forecast_data:
            _LOGGER.warning("No 7-day forecast available during 08:00 check for %s", self._zone)
            return

        # Pick the forecast with the maximum load
        max_forecast = max(forecast_data, key=lambda x: x["forecast_load_mw"])
        peak_forecast_load = max_forecast["forecast_load_mw"]
        peak_forecast_time = max_forecast["forecast_hour_ending"]

        # Store the forecast data (for debugging / attribute use) similar to update_forecast.
        self._forecast_data = forecast_data

        # Compare to the 5th highest known peak:
        fifth_peak = self._get_fifth_highest_peak()
        if peak_forecast_load >= 0.95 * fifth_peak:  # if it's close to or above the threshold
            self._high_risk_day = True
            _LOGGER.info("Flagging %s as high risk for new 5CP (forecast=%.1f, threshold=%.1f)", peak_forecast_load, fifth_peak)
        else:
            self._high_risk_day = False
            _LOGGER.info("Not a likely CP day for %s (forecast=%.1f, threshold=%.1f)", peak_forecast_load, fifth_peak)

    def _refine_peak_prediction(self, now_local, current_load, short_forecast_load):
        """
        Decide which hour is likely to be today's peak:
         1) Check official daily forecast peak hour
         2) Compare real-time load trends vs. forecast
         3) Possibly do rate-of-change or short regression to find if 
            the peak might shift from the official forecast hour 
         Returns (peak_load, peak_time).
        """
        # Basic approach: we expect the top load to occur late afternoon. 
        # Let's see if the short_forecast_load is higher than the 7-day forecasted peak for today.

        today = now_local.date()
        daily_forecast_peak = self._forecasted_daily_peaks.get(today, 0)
        # A simple bias correction:
        # If we consistently see a bias (rolling avg error), correct the daily peak forecast:
        avg_error = np.mean(self._rolling_forecast_errors) if self._rolling_forecast_errors else 0
        corrected_daily_peak = daily_forecast_peak + avg_error

        # Compare short vs corrected daily peak:
        if short_forecast_load is not None and short_forecast_load > corrected_daily_peak * 0.9:
            # If short-term forecast is close to or higher than daily forecast, trust short forecast.
            peak_load = (short_forecast_load + corrected_daily_peak) / 2
            peak_time = now_local + timedelta(hours=1)  # predict peak in about 1 hour
        else:
            peak_load = corrected_daily_peak
            # We don't have the *exact* hour from the 7-day data, so let's guess ~16-18 EPT for summer. 
            # For demonstration, we store ~17:00 local as the peak time:
            likely_peak_hour = datetime.combine(now_local.date(), time(17, 0)).astimezone(now_local.tzinfo)
            peak_time = likely_peak_hour

        # Rate-of-change approach if we are within 2 hours of that peak_time, 
        # we can do an optional quick check on the last few instantaneous loads 
        # to see if the load is ramping faster than expected:
        if peak_time - now_local <= timedelta(hours=2) and len(self._load_history) >= 5:
            # Some small polynomial fit or derivative check:
            # (Example for demonstration; real code might do a curve_fit)
            times, loads = self._extract_recent_history_arrays()
            # Just do a quick slope check:
            slope = np.polyfit(times, loads, 1)[0]
            if slope > 500:  # 500 MW per hour ramp is arbitrary example
                peak_load += slope * 1.0  # boost next-hour load by slope
            # ... or do a quadratic approach if desired ...

        return (int(round(peak_load)), peak_time)

    def _calculate_forecasted_rate_of_change(self, short_forecast_data):
        """Calculate rate of change (MW/hr) from short forecast."""
        if not short_forecast_data or len(short_forecast_data) < 2:
            return 0
        loads = [x["forecast_load_mw"] for x in short_forecast_data[:2]]
        times = [x["forecast_hour_ending"].timestamp() for x in short_forecast_data[:2]]
        delta_load = loads[1] - loads[0]
        delta_time = (times[1] - times[0]) / 3600  # hours
        return delta_load / delta_time if delta_time else 0

    def _calculate_instantaneous_rate_of_change(self):
        """Calculate instantaneous load rate from history (last hour)."""
        if len(self._load_history) < 2:
            return 0
        latest, oldest = self._load_history[-1], self._load_history[0]
        delta_load = latest[1] - oldest[1]
        delta_time = (latest[0] - oldest[0]).total_seconds() / 3600  # hours
        return delta_load / delta_time if delta_time else 0

    def _predict_time_shift(self, instantaneous_rate, forecasted_rate):
        """Predict shift in peak timing based on rate comparison."""
        rate_difference = instantaneous_rate - forecasted_rate
        if abs(rate_difference) < 100:  # Arbitrary threshold MW/hr
            return 0  # minimal shift
        # positive rate_difference means peak arriving sooner
        time_shift_minutes = -10 if rate_difference > 0 else 10
        return time_shift_minutes

    def _check_if_peak_hour_active(self, now_local):
        """
        True if we are currently in the predicted peak hour for a high-risk day.
        """
        if not self._predicted_peak_time:
            return False
        if self._state < self._get_fifth_highest_peak():
            return False
        # Round predicted peak time to the hour
        peak_hour_floor = self._predicted_peak_time.replace(minute=0, second=0, microsecond=0)
        return peak_hour_floor <= now_local < (peak_hour_floor + timedelta(hours=1))

    def _append_load_history(self, load):
        """
        Store load with its timestamp (UTC) in a rolling queue.
        """
        now_utc = datetime.now(timezone.utc)
        self._load_history.append((now_utc, load))

    def _update_actual_daily_peaks(self, now_local):
        """
        For each day in load_history, track the max load. 
        """
        today = now_local.date()
        # Filter loads from midnight local time to now
        today_midnight_utc = datetime.combine(today, time.min).astimezone(timezone.utc)
        daily_loads = [val for (ts, val) in self._load_history if ts >= today_midnight_utc]
        if daily_loads:
            self._actual_daily_peaks[today] = max(daily_loads)

    def _update_forecasted_daily_peaks(self, seven_day_forecast):
        """
        Extract each day's peak from the 7-day data. 
        """
        daily_peaks = defaultdict(int)
        daily_peak_times = {}
        for item in seven_day_forecast:
            dt_local = item["forecast_hour_ending"]
            load_mw = item["forecast_load_mw"]
            d = dt_local.date()
            if load_mw > daily_peaks[d]:
                daily_peaks[d] = load_mw
                daily_peak_times[d] = dt_local

        for d, load_val in daily_peaks.items():
            self._forecasted_daily_peaks[d] = load_val
            self._forecasted_daily_peak_time[d] = daily_peak_times[d]

    def _update_top_five_peaks(self):
        """
        Recompute top-5 from all known actual daily peaks and forecast daily peaks.
        """
        combined = list(self._actual_daily_peaks.values()) + list(self._forecasted_daily_peaks.values())
        # Keep only the top 5
        self._top_five_peaks = sorted(combined, reverse=True)[:5]

    def _get_fifth_highest_peak(self):
        """
        Return the 5th highest peak known, or the user threshold if fewer than 5.
        """
        if len(self._top_five_peaks) < 5:
            return self._user_defined_threshold
        return self._top_five_peaks[-1]

    def _record_forecast_error(self, actual_load):
        """
        Compare actual load vs. today's predicted peak. 
        If we're near or in the peak hour, or end-of-day, measure the difference.
        """
        if not self._predicted_peak_time:
            return

        # If it's near that peak hour, log an error:
        now_local = datetime.now(timezone.utc).astimezone()
        if abs((self._predicted_peak_time - now_local).total_seconds()) < 3600:
            error = actual_load - self._state  # positive if actual > predicted
            self._rolling_forecast_errors.append(error)

    def _compute_accuracy_probability(self):
        """
        Simplified approach: If forecast error is within +/- 5 GW, consider it 'accurate.'
        Return % of 'accurate' predictions in the rolling window.
        """
        if not self._rolling_forecast_errors:
            return "N/A"
        valid = [err for err in self._rolling_forecast_errors if abs(err) <= 5000]
        pct = len(valid) / len(self._rolling_forecast_errors) * 100
        return round(pct, 1)

    def _extract_recent_history_arrays(self):
        """
        Helper for polynomial fits. Return (times, loads) for the last N data points.
        times in hours from the earliest timestamp.
        """
        # sort by timestamp ascending
        sorted_hist = sorted(self._load_history, key=lambda x: x[0])
        base_time = sorted_hist[0][0]
        times = []
        loads = []
        for (ts, val) in sorted_hist:
            diff_hours = (ts - base_time).total_seconds() / 3600.0
            times.append(diff_hours)
            loads.append(val)
        return np.array(times), np.array(loads)

class PJMData:
    """Get and parse data from PJM with coordinated API rate limiting using your API key or fetched subscription key."""
    def __init__(self, websession, api_key):
        self._websession = websession
        self._subscription_key = api_key
        self._request_times = []
        self._lock = asyncio.Lock()

    async def _rate_limit(self):
        async with self._lock:
            now = time_module.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
            while len(self._request_times) >= 5:
                oldest = self._request_times[0]
                wait_time = 60 - (now - oldest) + 1
                _LOGGER.debug("API rate limit reached. Waiting %.2f seconds.", wait_time)
                await asyncio.sleep(wait_time)
                now = time_module.time()
                self._request_times = [t for t in self._request_times if now - t < 60]
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
        resource = "{}?{}".format(RESOURCE_INSTANTANEOUS, urllib.parse.urlencode(params))
        headers = self._get_headers()
        try:
            with async_timeout.timeout(60):
                response = await self._websession.get(resource, headers=headers)
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
            return None
        except Exception as err:
            _LOGGER.error("Unexpected error fetching load data: %s", err)
            return None

    async def async_update_forecast(self, zone):
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
        resource = "{}?{}".format(RESOURCE_FORECAST, urllib.parse.urlencode(params))
        headers = self._get_headers()
        try:
            with async_timeout.timeout(60):
                response = await self._websession.get(resource, headers=headers)
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
            return None
        except Exception as err:
            _LOGGER.error("Unexpected error fetching forecast data: %s", err)
            return None

    async def async_update_short_forecast(self, zone):
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
        resource = "{}?{}".format(RESOURCE_SHORT_FORECAST, urllib.parse.urlencode(params))
        headers = self._get_headers()
        try:
            with async_timeout.timeout(60):
                response = await self._websession.get(resource, headers=headers)
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
            return (None, None)
        except Exception as err:
            _LOGGER.error("Unexpected error fetching short forecast data: %s", err)
            return (None, None)

    async def async_update_lmp(self, pnode_id):
        await self._rate_limit()
        if not self._subscription_key:
            await self._get_subscription_key()
        now_utc = datetime.now(timezone.utc)
        current_minute = now_utc.minute
        if current_minute < 5:
            start_time_utc = (now_utc.replace(minute=4, second=0, microsecond=0) - timedelta(hours=1))
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
        resource = "{}?{}".format(RESOURCE_LMP, urllib.parse.urlencode(params))
        headers = self._get_headers()
        try:
            with async_timeout.timeout(60):
                response = await self._websession.get(resource, headers=headers)
                data = await response.json()
                if not data:
                    _LOGGER.error("No LMP data returned for pnode_id %s", pnode_id)
                    return None
                items = data["items"]
                total_lmp_values = [float(item["total_lmp_rt"]) for item in items if item["pnode_id"] == pnode_id]
                if not total_lmp_values:
                    _LOGGER.error("Couldn't find LMP data for pnode_id %s", pnode_id)
                    return None
                average_lmp = sum(total_lmp_values) / len(total_lmp_values)
                return round(average_lmp, 2)
        except (asyncio.TimeoutError, aiohttp.ClientError) as err:
            _LOGGER.error("Could not get LMP avg data from PJM: %s", err)
            return None
        except Exception as err:
            _LOGGER.error("Unexpected error fetching LMP avg data: %s", err)
            return None
