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

        if self._type in [CONF_TOTAL_LOAD_FORECAST, CONF_ZONE_LOAD_FORECAST, CONF_TOTAL_SHORT_FORECAST, CONF_ZONE_SHORT_FORECAST]:
            attr["forecast_data"] = self._forecast_data

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

    @Throttle(MIN_TIME_BETWEEN_UPDATES_FORECAST)
    async def update_forecast(self):
        forecast_data = await self._pjm_data.async_update_forecast(self._identifier)
        if forecast_data is not None:
            peak_forecast_load = max(forecast_data, key=lambda x: x["forecast_load_mw"])["forecast_load_mw"]
            self._state = peak_forecast_load
            self._forecast_data = forecast_data

    @Throttle(MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS)
    async def update_short_forecast(self):
        load, forecast_hour_ending = await self._pjm_data.async_update_short_forecast(self._identifier)
        if load is not None:
            self._state = load
        if forecast_hour_ending is not None:
            self._forecast_data = forecast_hour_ending

    @Throttle(MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS)
    async def update_lmp(self):
        lmp = await self._pjm_data.async_update_lmp(self._identifier)
        if lmp is not None:
            self._state = lmp

class CoincidentPeakPredictionSensor(SensorEntity):
    """
    This sensor predicts PJM (or zone) coincident peaks using a multi‑source
    forecasting approach with exponential smoothing. It uses:
    
      • The PJM Five Minute Load Forecast as a base,
      • Real‐time instantaneous load data to refine the forecast (via exponential smoothing),
      • A seven‑day forecast to provide a daily peak estimate.
      
    When within the 2‑hour window of the daily forecast peak, it also calculates the
    instantaneous rate of change and acceleration using a short rolling window and
    (optionally) applies a quadratic regression to dynamically adjust the prediction.
    
    The sensor maintains daily refined forecasts, updates historical peaks, and flags:
      - forecasted_peak_today (True if today’s predicted peak exceeds an effective threshold),
      - peak_hour_active (True if the current hour matches the predicted peak hour).
      
    Daily forecast data is reset at midnight, and historical peaks are reset on October 1st.
    """
    def __init__(self, pjm_data, zone, peak_threshold, accuracy_threshold, sensor_type):
        super().__init__()  # Initialize base SensorEntity
        self._pjm_data = pjm_data
        self._zone = zone
        self._sensor_type = sensor_type  # Either CONF_COINCIDENT_PEAK_PREDICTION_ZONE or _SYSTEM
        self._attr_name = f"Coincident Peak Prediction ({zone})"
        self._attr_unique_id = f"pjm_{sensor_type}_{zone}"
        self._unit_of_measurement = "MW"
        self._state = None

        # Configuration thresholds
        self._user_defined_threshold = peak_threshold
        self._accuracy_threshold = accuracy_threshold

        # Data storage (rolling history and forecast collections)
        self._load_history = deque(maxlen=MAX_HISTORY_SIZE)         # Instantaneous load history (~25 hours)
        self._daily_forecast = []                                   # List of tuples: (local_datetime, refined_load)
        self._actual_daily_peaks = {}                               # {date: peak_load} from actual loads
        self._forecasted_daily_peaks = {}                           # {date: peak_load} from seven-day forecast
        self._forecasted_daily_peak_time = {}                       # {date: peak_time} from seven-day forecast
        self._top_five_peaks = []                                   # Combined list of top five peaks
        self._historical_peak_accuracy = []                         # For accuracy tracking
        self._short_forecast_history = []                           # Store last few short-term forecasts

        # Extra attributes for peak prediction
        self._forecasted_peak_today = False
        self._peak_hour_active = False
        self._predicted_peak_time = None
        self._refined_forecast = None

    @property
    def icon(self):
            return "mdi:summit"

    @property
    def extra_state_attributes(self):
        return {
            "predicted_peak": self._state,
            "predicted_peak_time": self._predicted_peak_time.isoformat() if self._predicted_peak_time else None,
            "forecasted_peak_today": self._forecasted_peak_today,
            "peak_hour_active": self._peak_hour_active,
            "historical_peaks": list(self._actual_daily_peaks.values()) + list(self._forecasted_daily_peaks.values()),
            "top_five_peaks": self._top_five_peaks,
            "load_history": [(ts.isoformat(), load) for ts, load in self._load_history],
            "short_forecast_history": [(dt.isoformat(), load) for dt, load in self._short_forecast_history],
            "accuracy_probability_percent": round(self._compute_accuracy_probability() * 100, 1)
                if self._historical_peak_accuracy else "N/A",
        }

    async def async_update(self):
        """
        Update the sensor:
          1. Pull instantaneous load.
          2. Retrieve the short-term forecast (next 2 hours) and record it.
          3. Update refined forecast using exponential smoothing.
          4. Periodically (hourly) update the seven-day daily peak forecast.
          5. Update actual daily peaks from load history.
          6. Update top five peaks from combined historical and forecasted peaks.
          7. If within the 2-hour window of the daily forecast peak, compute rate-of-change and (optionally)
             apply quadratic regression to dynamically adjust the prediction.
          8. Blend the short-term forecast and any dynamic prediction.
          9. Set the sensor state and extra attributes accordingly.
        """
        # 1. Retrieve instantaneous load.
        load = await self._pjm_data.async_update_instantaneous(self._zone)
        if load is None:
            _LOGGER.error("No load data received for peak prediction")
            return
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone()
        self._load_history.append((now_utc, load))

        # Maintain a short history for rate-of-change calculations.
        if len(self._load_history) > 5:
            self._load_history.popleft()

        # 2. Retrieve short-term forecast (assumed to return a list of dicts with keys "load" and "timestamp")
        short_forecast_data = await self._pjm_data.async_update_short_forecast(self._zone)
        if short_forecast_data and isinstance(short_forecast_data, list):
            latest_forecast = short_forecast_data[0]
            short_peak_load = latest_forecast["load"]
            short_peak_time = datetime.fromtimestamp(latest_forecast["timestamp"], tz=timezone.utc).astimezone()
            # Record in short forecast history.
            self._short_forecast_history.append((short_peak_time, short_peak_load))
            if len(self._short_forecast_history) > 5:
                self._short_forecast_history.pop(0)
        else:
            short_peak_load = None
            short_peak_time = None

        # 3. Reset daily forecast if day has changed.
        if not self._daily_forecast or (self._daily_forecast and now_local.date() != self._daily_forecast[0][0].date()):
            self._daily_forecast = []
            # Initialize with short-term forecast if available.
            if short_peak_load is not None:
                self._refined_forecast = short_peak_load
                self._daily_forecast.append((short_peak_time, self._refined_forecast))
            else:
                self._refined_forecast = load

        # 4. Update refined forecast using exponential smoothing.
        alpha = 0.2
        if self._refined_forecast is None:
            self._refined_forecast = load
        else:
            self._refined_forecast = alpha * load + (1 - alpha) * self._refined_forecast
        self._daily_forecast.append((now_local, self._refined_forecast))

        # 5. Every hour, update the seven-day forecast.
        if now_local.minute == 0:
            seven_day_forecast = await self._pjm_data.async_update_forecast(self._zone)
            if seven_day_forecast:
                self._update_forecasted_daily_peaks(seven_day_forecast)

        # 6. Update actual daily peaks from load history.
        self._update_actual_daily_peaks(now_local)

        # 7. Update top five peaks.
        self._update_top_five_peaks()

        # 8. Determine predicted peak for today.
        today = now_local.date()
        # Get maximum refined forecast (from entries later than now) from today's forecast.
        refined_candidates = [(dt, val) for dt, val in self._daily_forecast if dt.date() == today and dt > now_local]
        if refined_candidates:
            st_time, st_load = max(refined_candidates, key=lambda x: x[1])
        else:
            st_time, st_load = None, 0

        # Retrieve daily forecast peak (from seven-day forecast) if available.
        df_load = self._forecasted_daily_peaks.get(today, 0)
        df_time = self._forecasted_daily_peak_time.get(today, None)

        # 9. Determine if we're inside the 2-hour window of the daily forecast peak.
        inside_two_hour = False
        if df_time and short_peak_time:
            if (df_time - now_local) < timedelta(hours=2):
                inside_two_hour = True

        # 10. If inside the 2-hour window, use instantaneous load dynamics.
        if inside_two_hour and len(self._load_history) >= 3:
            # Calculate rate-of-change and acceleration from the last few instantaneous load readings.
            times_arr = np.array([(ts - self._load_history[0][0]).total_seconds() / 3600.0 for ts, _ in self._load_history])
            loads_arr = np.array([val for _, val in self._load_history])
            dt = np.diff(times_arr)
            dload = np.diff(loads_arr)
            rate_of_change = dload / dt
            if len(rate_of_change) > 1:
                acceleration = np.diff(rate_of_change) / dt[:-1]
                avg_acceleration = np.mean(acceleration)
            else:
                avg_acceleration = 0

            # Optionally, use quadratic regression on recent data to project a peak.
            try:
                popt, _ = curve_fit(_quadratic, times_arr, loads_arr)
                a, b, c = popt
                if a < 0:
                    proj_peak_time_offset = -b / (2 * a)
                    proj_peak_load = _quadratic(proj_peak_time_offset, a, b, c)
                    # Convert offset to absolute datetime:
                    proj_peak_time = now_utc + timedelta(hours=proj_peak_time_offset)
                else:
                    proj_peak_time, proj_peak_load = None, None
            except Exception as exc:
                _LOGGER.error("Quadratic regression during 2-hr window failed: %s", exc)
                proj_peak_time, proj_peak_load = None, None

            # 11. Blending Logic: If the quadratic projection and short forecast differ,
            # blend them. Also, always prioritize short-term forecast inside the 2-hour window.
            if proj_peak_time and proj_peak_load and short_peak_load is not None:
                if abs(proj_peak_load - short_peak_load) > 500:
                    final_peak_time = proj_peak_time
                    final_peak_load = (proj_peak_load + short_peak_load) / 2
                else:
                    final_peak_time, final_peak_load = short_peak_time, short_peak_load
            else:
                final_peak_time, final_peak_load = short_peak_time, short_peak_load
        else:
            # Outside the 2-hour window, use daily forecast.
            final_peak_time, final_peak_load = df_time, df_load

        # 12. Update the sensor state.
        if final_peak_time is not None and final_peak_load:
            self._predicted_peak_time = final_peak_time
            self._state = round(final_peak_load)
        else:
            self._predicted_peak_time = None
            self._state = load

        # 13. Set flags based on an effective threshold.
        effective_threshold = max(self._user_defined_threshold, self._get_fifth_highest_peak())
        if self._predicted_peak_time and final_peak_load >= effective_threshold and (self._predicted_peak_time.date() == today):
            self._forecasted_peak_today = True
        else:
            self._forecasted_peak_today = False

        if self._predicted_peak_time:
            predicted_peak_hour = self._predicted_peak_time.replace(minute=0, second=0, microsecond=0)
            if final_peak_load >= effective_threshold and (predicted_peak_hour <= now_local < predicted_peak_hour + timedelta(hours=1)):
                self._peak_hour_active = True
            else:
                self._peak_hour_active = False
        else:
            self._peak_hour_active = False

        # 14. Update historical peaks and accuracy.
        self._update_historical_peaks(load)
        if now_local.minute < 5:
            accuracy = 1 if abs(load - final_peak_load) < 5000 else 0
            self._historical_peak_accuracy.append(accuracy)
            if len(self._historical_peak_accuracy) > 20:
                self._historical_peak_accuracy.pop(0)

    def _update_actual_daily_peaks(self, now_local):
        today = now_local.date()
        today_start = datetime.combine(today, time.min).astimezone(timezone.utc)
        todays_loads = [load for ts, load in self._load_history if ts >= today_start]
        if todays_loads:
            self._actual_daily_peaks[today] = max(todays_loads)
            _LOGGER.debug("Updated actual daily peak for %s: %s MW", today, self._actual_daily_peaks[today])

    def _update_forecasted_daily_peaks(self, seven_day_forecast):
        daily_peaks = defaultdict(list)
        daily_peak_times = {}
        for item in seven_day_forecast:
            dt = item["forecast_hour_ending"].date()
            load = item["forecast_load_mw"]
            daily_peaks[dt].append(load)
            if dt not in daily_peak_times or load > daily_peak_times[dt][1]:
                daily_peak_times[dt] = (item["forecast_hour_ending"], load)
        for dt, loads in daily_peaks.items():
            self._forecasted_daily_peaks[dt] = max(loads)
        self._forecasted_daily_peak_time = {dt: daily_peak_times[dt][0] for dt in daily_peak_times}
        _LOGGER.debug("Updated forecasted daily peaks: %s", self._forecasted_daily_peaks)

    def _update_top_five_peaks(self):
        combined = list(self._actual_daily_peaks.values()) + list(self._forecasted_daily_peaks.values())
        self._top_five_peaks = sorted(combined, reverse=True)[:5]
        _LOGGER.debug("Updated top five peaks: %s", self._top_five_peaks)

    def _get_fifth_highest_peak(self):
        if len(self._top_five_peaks) < 5:
            return self._user_defined_threshold
        return self._top_five_peaks[-1]

    def _update_historical_peaks(self, load):
        if load > self._user_defined_threshold:
            self._historical_peaks.append(load)
            self._historical_peaks = sorted(self._historical_peaks, reverse=True)[:5]
            _LOGGER.debug("Historical peaks updated: %s", self._historical_peaks)

    def _compute_accuracy_probability(self):
        if not self._historical_peak_accuracy:
            return 1.0
        return sum(self._historical_peak_accuracy) / len(self._historical_peak_accuracy)


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
                    forecast_hour_ending = datetime.strptime(item['forecast_datetime_ending_utc'], '%Y-%m-%dT%H:%M:%S') \
                        .replace(tzinfo=timezone.utc).astimezone()
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
                sorted_data = sorted(
                    data, 
                    key=lambda x: (
                        x["forecast_load_mw"] * -1, 
                        datetime.strptime(x['forecast_datetime_ending_utc'],'%Y-%m-%dT%H:%M:%S')
                        .replace(tzinfo=timezone.utc).astimezone()
                    )
                )
                if sorted_data:
                    load = int(sorted_data[0]["forecast_load_mw"])
                    forecast_hour_ending = datetime.strptime(sorted_data[0]['forecast_datetime_ending_utc'],
                        '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone()
                    unix_time = forecast_hour_ending.timestamp()
                    return (load, unix_time)
                return (None, None)
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
