"""
PJM Sensor Integration
----------------------

This module provides support for multiple PJM sensors—including the brand
new Coincident Peak Prediction sensor that uses real-time load trends,
derivative analysis, and quadratic regression to predict coincident peaks
at the start of the hour. All API calls share the same rate-limited PJMData instance.

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
    CONF_INSTANTANEOUS_ZONE_LOAD,
    CONF_INSTANTANEOUS_TOTAL_LOAD,
    CONF_ZONE_LOAD_FORECAST,
    CONF_TOTAL_LOAD_FORECAST,
    CONF_ZONE_SHORT_FORECAST,
    CONF_TOTAL_SHORT_FORECAST,
    CONF_ZONAL_LMP,
    CONF_COINCIDENT_PEAK_PREDICTION,
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
RESOURCE_SUBSCRIPTION_KEY = 'https://dataminer2.pjm.com/config/settings.json'

MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS = timedelta(seconds=300)  # 5 minutes for load, LMPs
MIN_TIME_BETWEEN_UPDATES_FORECAST = timedelta(seconds=3600)  # 1 hour for forecasts

PJM_RTO_ZONE = "PJM RTO"
FORECAST_COMBINED_ZONE = 'RTO_COMBINED'

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback):
    """Set up the PJM sensor platform from config entry."""
    zone = entry.data["zone"]
    selected_sensors = entry.data["sensors"]
    pjm_data = PJMData(async_get_clientsession(hass))
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
        elif sensor_type == CONF_COINCIDENT_PEAK_PREDICTION:
            # Use user-provided thresholds (if any) or fall back to defaults.
            peak_threshold = entry.data.get(CONF_PEAK_THRESHOLD, DEFAULT_PEAK_THRESHOLD)
            accuracy_threshold = entry.data.get(CONF_ACCURACY_THRESHOLD, DEFAULT_ACCURACY_THRESHOLD)
            dev.append(CoincidentPeakPredictionSensor(pjm_data, zone, peak_threshold, accuracy_threshold))
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
    
    It uses real-time load data (fetched via the shared PJMData instance), tracks the load history,
    performs derivative analysis, and fits a quadratic regression to predict the peak load and its arrival time.
    
    Only if the predicted peak load exceeds the threshold (the higher of the user-defined or the 5th highest historical peak)
    and our rolling prediction accuracy is above the provided accuracy threshold will a peak be flagged.
    """
    def __init__(self, pjm_data, zone, peak_threshold, accuracy_threshold):
        self._pjm_data = pjm_data
        self._zone = zone
        self._attr_name = f"Coincident Peak Prediction ({zone})"
        self._attr_unique_id = f"pjm_{CONF_COINCIDENT_PEAK_PREDICTION}_{zone}"
        self._unit_of_measurement = "MW"
        self._state = None

        # Configuration thresholds
        self._user_defined_threshold = peak_threshold
        self._accuracy_threshold = accuracy_threshold

        # Keep recent load history (timestamp, load) and historical data
        self._load_history = deque(maxlen=50)
        self._historical_peaks = []
        self._historical_peak_accuracy = []

    @property
    def name(self):
        return self._attr_name

    @property
    def unique_id(self):
        return self._attr_unique_id

    @property
    def unit_of_measurement(self):
        return self._unit_of_measurement

    @property
    def native_value(self):
        return self._state

    @property
    def extra_state_attributes(self):
        return {
            "predicted_peak": self._state,
            "accuracy_probability_percent": round(self._compute_accuracy_probability() * 100, 1),
            "load_history": [(ts.isoformat(), load) for ts, load in self._load_history],
            "historical_peaks": self._historical_peaks,
        }

    @Throttle(MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS)
    async def async_update(self):
        # 1. Fetch the current instantaneous load (yes, we’re using the same delicious API call!)
        load = await self._pjm_data.async_update_instantaneous(self._zone)
        if load is None:
            _LOGGER.error("No load data received for peak prediction—oh, the nerve!")
            return

        now = datetime.now(timezone.utc)
        self._load_history.append((now, load))
        _LOGGER.debug("Peak predictor: added load %s MW at %s", load, now.isoformat())

        # Wait until we have enough data points
        if len(self._load_history) < 5:
            self._state = load
            return

        # 2. Find the most recent local minimum (the moment the load started its sultry ascent)
        local_min_index = self._find_local_minimum_index()
        if local_min_index is None:
            _LOGGER.debug("No local minimum detected yet. More data, please, my dear!")
            self._state = load
            return

        # 3. Build regression data from the local minimum to now
        times = []
        loads = []
        t0 = self._load_history[local_min_index][0]
        for t, l in list(self._load_history)[local_min_index:]:
            delta_minutes = (t - t0).total_seconds() / 60.0
            times.append(delta_minutes)
            loads.append(l)

        if len(times) < 3:
            _LOGGER.debug("Not enough data for a proper quadratic fit. Patience, sexy!")
            self._state = load
            return

        try:
            # Fit quadratic: load = A*t^2 + B*t + C
            coeffs = np.polyfit(times, loads, 2)
            A, B, C = coeffs
            _LOGGER.debug("Quadratic fit coefficients: A=%.4f, B=%.4f, C=%.4f", A, B, C)
        except Exception as e:
            _LOGGER.error("Quadratic regression failed: %s", e)
            self._state = load
            return

        # 4. Check that the curve is concave down (A < 0) so a peak can exist
        if A >= 0:
            _LOGGER.debug("Curve not concave down (A=%.4f). No peak forming, darling.", A)
            self._state = load
            return

        # 5. Calculate the time (in minutes from t0) when the derivative is zero (i.e. the peak)
        t_peak = -B / (2 * A)
        if t_peak <= times[-1]:
            _LOGGER.debug("Predicted peak time (%.2f minutes) already passed. Time to focus on the future!", t_peak)
            self._state = load
            return

        predicted_peak_load = A * t_peak**2 + B * t_peak + C
        _LOGGER.debug("Predicted peak load: %.2f MW arriving in %.2f minutes", predicted_peak_load, t_peak)

        # 6. Use the higher of the user-defined threshold or the 5th highest historical peak
        threshold = max(self._user_defined_threshold, self._get_fifth_highest_peak())
        _LOGGER.debug("Effective threshold used: %.2f MW", threshold)
        if predicted_peak_load < threshold:
            _LOGGER.debug("Predicted peak (%.2f MW) is below threshold. Not sizzling enough!", predicted_peak_load)
            self._state = load
            return

        # 7. Ensure our historical prediction accuracy is up to snuff
        accuracy_prob = self._compute_accuracy_probability()
        _LOGGER.debug("Prediction accuracy probability: %.2f%%", accuracy_prob * 100)
        if accuracy_prob < self._accuracy_threshold:
            _LOGGER.debug("Accuracy probability (%.2f%%) is below the threshold of %.2f%%. Need more practice, baby!", accuracy_prob * 100, self._accuracy_threshold * 100)
            self._state = load
            return

        # 8. Check for load flattening (extended peak)
        t_last, load_last = self._load_history[-1]
        t_prev, load_prev = self._load_history[-2]
        dt_minutes = (t_last - t_prev).total_seconds() / 60.0 or 1
        derivative = (load_last - load_prev) / dt_minutes
        _LOGGER.debug("Current derivative: %.2f MW/min", derivative)

        now_local = datetime.now()
        if now_local.minute < 5:
            if load >= 0.95 * predicted_peak_load and abs(derivative) <= 10:
                self._state = f"PEAK (extended): {round(predicted_peak_load)} MW"
                _LOGGER.info("Extended peak flagged: %s", self._state)
            else:
                peak_time_est = (t0 + timedelta(minutes=t_peak)).strftime('%H:%M')
                self._state = f"PEAK predicted: {round(predicted_peak_load)} MW at {peak_time_est}"
                _LOGGER.info("Peak predicted: %s", self._state)
        else:
            self._state = load

        # 9. Update historical peaks
        self._update_historical_peaks(load)

        # 10. Update prediction accuracy at the hour’s start (simulate a hit if within 5,000 MW)
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
    """Get and parse data from PJM with coordinated API rate limiting."""

    def __init__(self, websession):
        self._websession = websession
        self._subscription_key = None
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
        if self._subscription_key:
            return
        try:
            with async_timeout.timeout(60):
                response = await self._websession.get(RESOURCE_SUBSCRIPTION_KEY)
                data = await response.json()
                self._subscription_key = data['subscriptionKey']
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
