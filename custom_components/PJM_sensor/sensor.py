"""
PJM Sensor Integration
----------------------

This module provides support for multiple PJM sensors—including the brand
new Coincident Peak Prediction sensor that uses real-time load trends,
derivative analysis, and quadratic regression to predict coincident peaks
at the start of the hour. All API calls now share your provided, tantalizing API key.

Enjoy the passion—and the power—of smart energy monitoring!
"""

import asyncio
from collections import deque
from datetime import datetime, date, time, timezone, timedelta
import logging
import urllib.parse
import time as time_module

import async_timeout
import aiohttp
import numpy as np

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
    DEFAULT_PEAK_THRESHOLD,
    DEFAULT_ACCURACY_THRESHOLD,
    ZONE_TO_PNODE_ID,
    SENSOR_TYPES,
)

_LOGGER = logging.getLogger(__name__)

RESOURCE_INSTANTANEOUS = 'https://api.pjm.com/api/v1/inst_load'
RESOURCE_FORECAST = 'https://api.pjm.com/api/v1/load_frcstd_7_day'
RESOURCE_SHORT_FORECAST = 'https://api.pjm.com/api/v1/very_short_load_frcst'
RESOURCE_LMP = 'https://api.pjm.com/api/v1/rt_unverified_fivemin_lmps'
# The subscription key resource is no longer needed since we use the provided API key.
# RESOURCE_SUBSCRIPTION_KEY = 'https://dataminer2.pjm.com/config/settings.json'

MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS = timedelta(seconds=300)  # 5 minutes for load, LMPs
MIN_TIME_BETWEEN_UPDATES_FORECAST = timedelta(seconds=3600)  # 1 hour for forecasts

PJM_RTO_ZONE = "PJM RTO"
FORECAST_COMBINED_ZONE = 'RTO_COMBINED'

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback):
    """Set up the PJM sensor platform from a config entry."""
    zone = entry.data["zone"]
    selected_sensors = entry.data["sensors"]
    # Pass the provided API key to PJMData!
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
        elif sensor_type == CONF_COINCIDENT_PEAK_PREDICTION_ZONE:
            peak_threshold = entry.data.get(CONF_PEAK_THRESHOLD, DEFAULT_PEAK_THRESHOLD)
            accuracy_threshold = entry.data.get(CONF_ACCURACY_THRESHOLD, DEFAULT_ACCURACY_THRESHOLD)
            dev.append(CoincidentPeakPredictionSensor(
                pjm_data, zone, peak_threshold, accuracy_threshold, CONF_COINCIDENT_PEAK_PREDICTION_ZONE))
        elif sensor_type == CONF_COINCIDENT_PEAK_PREDICTION_SYSTEM:
            peak_threshold = entry.data.get(CONF_PEAK_THRESHOLD, DEFAULT_PEAK_THRESHOLD)
            accuracy_threshold = entry.data.get(CONF_ACCURACY_THRESHOLD, DEFAULT_ACCURACY_THRESHOLD)
            dev.append(CoincidentPeakPredictionSensor(
                pjm_data, PJM_RTO_ZONE, peak_threshold, accuracy_threshold, CONF_COINCIDENT_PEAK_PREDICTION_SYSTEM))
        else:
            dev.append(PJMSensor(pjm_data, sensor_type, identifier, None))

    async_add_entities(dev, True)

    # Schedule staggered updates using async tasks
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
    This sensor predicts PJM system coincident peaks at the start of the hour.
    
    It uses real-time load data (via the shared PJMData instance), tracks a rolling load history,
    and performs quadratic regression to forecast the peak load and its arrival time.
    
    In addition, it now provides:
      - A boolean attribute 'forecasted_peak_today' that is true all day if a valid forecast
        predicts that the peak (above the effective threshold) will occur today.
      - A boolean attribute 'peak_hour_active' that remains true for the entire hour when the predicted peak occurs.
    
    The historical peaks list (top 5) is reset on October 1st.
    """
    def __init__(self, pjm_data, zone, peak_threshold, accuracy_threshold, sensor_type):
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

        # Increase the load history to 300 data points (approx. 25 hours)
        self._load_history = deque(maxlen=300)
        self._historical_peaks = []
        self._historical_peak_accuracy = []

        # New attributes for peak prediction
        self._forecasted_peak_today = False
        self._peak_hour_active = False
        self._predicted_peak_time = None

    @property
    def extra_state_attributes(self):
        return {
            "predicted_peak": self._state,
            "predicted_peak_time": self._predicted_peak_time.isoformat() if self._predicted_peak_time else None,
            "forecasted_peak_today": self._forecasted_peak_today,
            "peak_hour_active": self._peak_hour_active,
            "accuracy_probability_percent": round(self._compute_accuracy_probability() * 100, 1) if self._historical_peak_accuracy else "N/A",
            "load_history": [(ts.isoformat(), load) for ts, load in self._load_history],
            "historical_peaks": self._historical_peaks,
        }

    async def async_update(self):
        # 1. Retrieve current load
        load = await self._pjm_data.async_update_instantaneous(self._zone)
        if load is None:
            _LOGGER.error("No load data received for peak prediction")
            return

        now_utc = datetime.now(timezone.utc)
        self._load_history.append((now_utc, load))
        _LOGGER.debug("Peak predictor: added load %s MW at %s", load, now_utc.isoformat())

        # Reset historical peaks on October 1st (local time)
        now_local = datetime.now().astimezone()
        if now_local.month == 10 and now_local.day == 1:
            self._historical_peaks = []
            _LOGGER.debug("Historical peaks reset on October 1st.")

        # If insufficient data (e.g. fewer than 12 points), do not predict yet.
        if len(self._load_history) < 12:
            self._state = load  # just show current load
            return

        # 2. Find the most recent local minimum ("turning point")
        local_min_index = self._find_local_minimum_index()
        if local_min_index is None:
            _LOGGER.debug("No local minimum detected in the current load history.")
            self._state = load  # do not predict yet
            return

        # 3. Build regression dataset from the local minimum to now
        times = []
        loads = []
        t0 = self._load_history[local_min_index][0]
        for t, l in list(self._load_history)[local_min_index:]:
            delta_minutes = (t - t0).total_seconds() / 60.0
            times.append(delta_minutes)
            loads.append(l)
        if len(times) < 3:
            _LOGGER.debug("Not enough data for quadratic regression.")
            self._state = load
            return

        try:
            # Fit a quadratic: load = A*t^2 + B*t + C
            coeffs = np.polyfit(times, loads, 2)
            A, B, C = coeffs
            _LOGGER.debug("Quadratic coefficients: A=%.4f, B=%.4f, C=%.4f", A, B, C)
        except Exception as e:
            _LOGGER.error("Quadratic regression failed: %s", e)
            self._state = load
            return

        # 4. Ensure the curve is concave down (A < 0)
        if A >= 0:
            _LOGGER.debug("Curve not concave down (A=%.4f). No peak forming.", A)
            self._state = load
            return

        # 5. Calculate predicted time (in minutes from t0) when the derivative is zero (i.e. the peak)
        t_peak = -B / (2 * A)
        if t_peak <= times[-1]:
            _LOGGER.debug("Predicted peak time (%.2f minutes) already passed.", t_peak)
            self._state = load
            return

        predicted_peak_load = A * t_peak**2 + B * t_peak + C
        _LOGGER.debug("Predicted peak load: %.2f MW, calculated from t_peak=%.2f minutes", predicted_peak_load, t_peak)

        # 6. Determine the effective threshold (the higher of user threshold and the 5th historical peak)
        effective_threshold = max(self._user_defined_threshold, self._get_fifth_highest_peak())
        _LOGGER.debug("Effective threshold used: %.2f MW", effective_threshold)
        if predicted_peak_load < effective_threshold:
            _LOGGER.debug("Predicted peak (%.2f MW) is below the effective threshold.", predicted_peak_load)
            self._state = load
            return

        # 7. Compute the predicted peak datetime (using the t0 from the local minimum)
        predicted_peak_time = t0 + timedelta(minutes=t_peak)
        # Convert to local time for comparison
        self._predicted_peak_time = predicted_peak_time.astimezone()

        # 8. Set the 'forecasted_peak_today' attribute:
        if self._predicted_peak_time.date() == now_local.date():
            self._forecasted_peak_today = True
        else:
            self._forecasted_peak_today = False

        # 9. Set the 'peak_hour_active' attribute:
        predicted_peak_hour = self._predicted_peak_time.replace(minute=0, second=0, microsecond=0)
        if predicted_peak_hour <= now_local < (predicted_peak_hour + timedelta(hours=1)):
            self._peak_hour_active = True
        else:
            self._peak_hour_active = False

        # 10. Update the sensor state:
        if now_local.minute < 5:
            self._state = f"PEAK predicted: {round(predicted_peak_load)} MW at {self._predicted_peak_time.strftime('%H:%M')}"
            _LOGGER.info("Peak predicted: %s", self._state)
        else:
            self._state = load

        # 11. Update historical peaks if this load qualifies
        self._update_historical_peaks(load)

        # 12. Update historical prediction accuracy (if within the first 5 minutes of the hour)
        if now_local.minute < 5:
            accuracy = 1 if abs(load - predicted_peak_load) < 5000 else 0
            self._historical_peak_accuracy.append(accuracy)
            if len(self._historical_peak_accuracy) > 20:
                self._historical_peak_accuracy.pop(0)

    def _find_local_minimum_index(self):
        data = list(self._load_history)
        for i in range(len(data) - 2, 0, -1):
            _, load_prev = data[i - 1]
            _, load_curr = data[i]
            _, load_next = data[i + 1]
            if (load_curr - load_prev) < 0 and (load_next - load_curr) > 0:
                return i
        return None

    def _get_fifth_highest_peak(self):
        if len(self._historical_peaks) < 5:
            return self._user_defined_threshold
        sorted_peaks = sorted(self._historical_peaks, reverse=True)
        return sorted_peaks[4]

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
    """Get and parse data from PJM with coordinated API rate limiting using your API key."""
    def __init__(self, websession, api_key):
        self._websession = websession
        self._subscription_key = api_key  # Use the provided API key
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
        headers = {
            'Ocp-Apim-Subscription-Key': self._subscription_key,
            'Content-Type': 'application/json',
        }
        return headers

    async def _get_subscription_key(self):
        # Since an API key is provided via configuration, we simply ensure it's set.
        if self._subscription_key:
            return
        _LOGGER.error("API key not provided, cannot proceed with API calls.")

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
                sorted_data = sorted(
                    data, 
                    key=lambda x: (
                        x["forecast_load_mw"] * -1, 
                        datetime.strptime(x['forecast_datetime_ending_utc'],'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone()
                    )
                )
                if sorted_data:
                    load = int(sorted_data[0]["forecast_load_mw"])
                    forecast_hour_ending = datetime.strptime(sorted_data[0]['forecast_datetime_ending_utc'],'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone()
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
