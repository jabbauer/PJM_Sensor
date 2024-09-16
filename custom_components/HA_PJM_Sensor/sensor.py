"""
Support for PJM data.
"""
import asyncio
from datetime import datetime as dt, date, time, timezone, timedelta
import logging

import aiohttp
import async_timeout
import voluptuous as vol
import urllib.parse

from homeassistant.helpers.entity import Entity
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import Throttle
from homeassistant.const import (
    CONF_API_KEY,
    CONF_NAME,
    CONF_ZONE,
    CONF_SCAN_INTERVAL,
)

from .const import (
    DOMAIN,
    SENSOR_TYPES,
    ZONE_TO_PNODE_ID,
    CONF_ZONAL_LMP_AVG,
    CONF_ZONAL_LMP_5MIN,
    CONF_INSTANTANEOUS_ZONE_LOAD,
    CONF_INSTANTANEOUS_TOTAL_LOAD,
    CONF_ZONE_LOAD_FORECAST,
    CONF_TOTAL_LOAD_FORECAST,
    CONF_ZONE_SHORT_FORECAST,
    CONF_TOTAL_SHORT_FORECAST,
)

_LOGGER = logging.getLogger(__name__)

RESOURCE_INSTANTANEOUS = 'https://api.pjm.com/api/v1/inst_load'
RESOURCE_SHORT_FORECAST = 'https://api.pjm.com/api/v1/very_short_load_frcst'
RESOURCE_FORECAST = 'https://api.pjm.com/api/v1/load_frcstd_7_day'
RESOURCE_LMP = 'https://api.pjm.com/api/v1/rt_unverified_fivemin_lmps'

# Default update frequencies
DEFAULT_SCAN_INTERVAL_INSTANTANEOUS = 300  # seconds
DEFAULT_SCAN_INTERVAL_FORECAST = 3600  # seconds

ICON_POWER = 'mdi:flash'

MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS = timedelta(seconds=300)
MIN_TIME_BETWEEN_UPDATES_FORECAST = timedelta(seconds=3600)

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up PJM sensors from a config entry."""
    config = config_entry.data
    subscription_key = config.get(CONF_API_KEY)
    update_frequency = config.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL_INSTANTANEOUS)

    pjm_data = PJMData(async_get_clientsession(hass), subscription_key, update_frequency)
    # Removed: await pjm_data.async_update()

    sensors = []
    monitored_variables = config.get(CONF_MONITORED_VARIABLES, [])

    for variable in monitored_variables:
        sensor_type = variable['type']
        zone = variable.get(CONF_ZONE)
        name = variable.get(CONF_NAME)

        if sensor_type in (
            CONF_INSTANTANEOUS_TOTAL_LOAD,
            CONF_ZONE_LOAD_FORECAST,
            CONF_TOTAL_LOAD_FORECAST,
            CONF_ZONE_SHORT_FORECAST,
            CONF_TOTAL_SHORT_FORECAST,
            CONF_ZONAL_LMP_AVG,
            CONF_ZONAL_LMP_5MIN,
        ):
            if sensor_type in (CONF_ZONAL_LMP_AVG, CONF_ZONAL_LMP_5MIN):
                pnode_id = ZONE_TO_PNODE_ID.get(zone)
                if not pnode_id:
                    _LOGGER.error("Invalid zone provided for LMP: %s", zone)
                    continue
                sensors.append(PJMSensor(pjm_data, sensor_type, pnode_id, name))
            else:
                sensors.append(PJMSensor(pjm_data, sensor_type, zone, name))
        else:
            _LOGGER.error("Unknown sensor type: %s", sensor_type)

    async_add_entities(sensors, True)

class PJMSensor(Entity):
    """Implementation of a PJM sensor."""

    def __init__(self, pjm_data, sensor_type, identifier, name):
        """Initialize the sensor."""
        self._pjm_data = pjm_data
        self._type = sensor_type
        self._identifier = identifier
        self._name = name if name else SENSOR_TYPES[sensor_type][0]
        self._unit_of_measurement = SENSOR_TYPES[sensor_type][1]
        self._state = None
        self._forecast_data = None
        self._unique_id = f"{DOMAIN}_{sensor_type}_{identifier}"

        # Customize the name based on sensor type
        if sensor_type == CONF_ZONAL_LMP_AVG:
            zone_name = next((zone for zone, id in ZONE_TO_PNODE_ID.items() if id == identifier), "Unknown Zone")
            self._name = f"{zone_name} {SENSOR_TYPES[sensor_type][0]}"
        elif sensor_type == CONF_ZONAL_LMP_5MIN:
            zone_name = next((zone for zone, id in ZONE_TO_PNODE_ID.items() if id == identifier), "Unknown Zone")
            self._name = f"{zone_name} {SENSOR_TYPES[sensor_type][0]}"

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name
    
    @property
    def unique_id(self):
        """Return a unique ID for the sensor."""
        return self._unique_id

    @property
    def icon(self):
        """Icon to use in the frontend, if any."""
        return ICON_POWER

    @property
    def state(self):
        """Return the state of the sensor."""
        return self._state

    @property
    def unit_of_measurement(self):
        """Return the unit of measurement of this entity, if any."""
        return self._unit_of_measurement

    @property
    def extra_state_attributes(self):
        """Return the state attributes."""
        attr = {}
        if self._identifier and self._type not in [CONF_TOTAL_LOAD_FORECAST, CONF_TOTAL_SHORT_FORECAST]:
            attr["identifier"] = self._identifier

        if self._type in [CONF_TOTAL_LOAD_FORECAST, CONF_ZONE_LOAD_FORECAST, CONF_TOTAL_SHORT_FORECAST, CONF_ZONE_SHORT_FORECAST]:
            attr["forecast_data"] = self._forecast_data

        return attr

    async def async_update(self):
        """Use the PJM data to set our state."""
        try:
            if self._type in [CONF_INSTANTANEOUS_ZONE_LOAD, CONF_INSTANTANEOUS_TOTAL_LOAD]:
                await self.update_load()
            elif self._type == CONF_ZONAL_LMP_AVG:
                await self.update_lmp_avg()
            elif self._type == CONF_ZONAL_LMP_5MIN:
                await self.update_lmp_5min()
            elif self._type in [CONF_TOTAL_SHORT_FORECAST, CONF_ZONE_SHORT_FORECAST]:
                await self.update_short_forecast()
            else:
                await self.update_forecast()
        except (ValueError, KeyError):
            _LOGGER.error("Could not update status for %s", self._name)
        except AttributeError as err:
            _LOGGER.error("Could not update status for PJM: %s", err)
        except TypeError:
            # Possibly throttled update; ignore
            pass
        except Exception as err:
            _LOGGER.error("Unknown error for PJM: %s", err)

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
    async def update_lmp_avg(self):
        lmp_avg = await self._pjm_data.async_update_lmp_avg(self._identifier)
        if lmp_avg is not None:
            self._state = lmp_avg

    @Throttle(MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS)
    async def update_lmp_5min(self):
        lmp_5min = await self._pjm_data.async_update_lmp_5min(self._identifier)
        if lmp_5min is not None:
            self._state = lmp_5min

class PJMData:
    """Get and parse data from PJM."""

    def __init__(self, websession, subscription_key, scan_interval):
        """Initialize the data object."""
        self._websession = websession
        self._subscription_key = subscription_key
        self._scan_interval = scan_interval

    def _get_headers(self):
        headers = {
            'Ocp-Apim-Subscription-Key': self._subscription_key,
            'Content-Type': 'application/json',
        }
        return headers

    async def async_update_instantaneous(self, zone):
        """Fetch instantaneous load data."""
        params = {
            'rowCount': '100',
            'sort': 'datetime_beginning_utc',
            'order': 'Desc',
            'startRow': '1',
            'isActiveMetadata': 'true',
            'fields': 'area,instantaneous_load',
            'datetime_beginning_utc': self._get_time_range(minutes=10),
        }
        resource = f"{RESOURCE_INSTANTANEOUS}?{urllib.parse.urlencode(params)}"
        headers = self._get_headers()

        try:
            with async_timeout.timeout(60):
                response = await self._websession.get(resource, headers=headers)
                data = await response.json()

                if not data:
                    _LOGGER.error("No load data returned for zone %s", zone)
                    return None

                items = data.get("items", [])

                for item in items:
                    if item.get("area") == zone:
                        return int(round(item.get("instantaneous_load", 0)))

                _LOGGER.error("Couldn't find load data for zone %s", zone)
                return None

        except (asyncio.TimeoutError, aiohttp.ClientError) as err:
            _LOGGER.error("Could not get load data from PJM: %s", err)
            return None
        except Exception as err:
            _LOGGER.error("Unexpected error fetching load data: %s", err)
            return None

    async def async_update_forecast(self, zone):
        """Fetch load forecast data."""
        midnight_local = dt.combine(date.today(), time())
        start_time_utc = midnight_local.astimezone(timezone.utc)
        end_time_utc = start_time_utc + timedelta(hours=23, minutes=59)
        time_string = self._format_time_range(start_time_utc, end_time_utc)

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
                full_data = await response.json()
                data = full_data.get("items", [])

                forecast_data = []
                for item in data:
                    forecast_hour_ending = dt.strptime(item['forecast_datetime_ending_utc'], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone()
                    forecast_data.append({
                        "forecast_hour_ending": forecast_hour_ending,
                        "forecast_load_mw": int(item.get("forecast_load_mw", 0))
                    })

                return forecast_data

        except (asyncio.TimeoutError, aiohttp.ClientError) as err:
            _LOGGER.error("Could not get forecast data from PJM: %s", err)
            return None
        except Exception as err:
            _LOGGER.error("Unexpected error fetching forecast data: %s", err)
            return None

    async def async_update_short_forecast(self, zone):
        """Fetch short-term load forecast data."""
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
                full_data = await response.json()
                data = full_data.get("items", [])

                sorted_data = sorted(
                    data,
                    key=lambda x: (-x.get("forecast_load_mw", 0), 
                                  dt.strptime(x['forecast_datetime_ending_utc'], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone())
                )

                if not sorted_data:
                    _LOGGER.error("No short forecast data available for zone %s", zone)
                    return (None, None)

                top_item = sorted_data[0]
                load = int(top_item.get("forecast_load_mw", 0))
                forecast_hour_ending = dt.strptime(top_item['forecast_datetime_ending_utc'], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone()
                return (load, forecast_hour_ending)

        except (asyncio.TimeoutError, aiohttp.ClientError) as err:
            _LOGGER.error("Could not get short forecast data from PJM: %s", err)
            return (None, None)
        except Exception as err:
            _LOGGER.error("Unexpected error fetching short forecast data: %s", err)
            return (None, None)

    async def async_update_lmp_avg(self, pnode_id):
        """Fetch hourly average LMP data."""
        now_utc = dt.now(timezone.utc)
        start_time_utc = now_utc - timedelta(hours=1)
        time_string = self._format_time_range(start_time_utc, now_utc)

        params = {
            'rowCount': '60',
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
                data = await response.json()

                if not data:
                    _LOGGER.error("No LMP data returned for pnode_id %s", pnode_id)
                    return None

                items = data.get("items", [])

                total_lmp_values = [float(item.get("total_lmp_rt", 0)) for item in items if item.get("pnode_id") == pnode_id]
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

    async def async_update_lmp_5min(self, pnode_id):
        """Fetch 5-minute LMP data."""
        now_utc = dt.now(timezone.utc)
        current_minute = now_utc.minute
        if current_minute < 5:
            start_time_utc = (now_utc.replace(minute=0, second=0, microsecond=0) - timedelta(minutes=5))
        else:
            start_time_utc = now_utc.replace(minute=(current_minute // 5) * 5, second=0, microsecond=0)
        time_string = self._format_time_range(start_time_utc, now_utc)

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
                data = await response.json()

                if not data:
                    _LOGGER.error("No LMP data returned for pnode_id %s", pnode_id)
                    return None

                items = data.get("items", [])
                lmp_values = [float(item.get("total_lmp_rt", 0)) for item in items if item.get("pnode_id") == pnode_id]

                if not lmp_values:
                    _LOGGER.error("Couldn't find LMP data for pnode_id %s", pnode_id)
                    return None

                latest_lmp = lmp_values[-1]  # Get the most recent 5-min LMP
                return round(latest_lmp, 2)

        except (asyncio.TimeoutError, aiohttp.ClientError) as err:
            _LOGGER.error("Could not get LMP 5-min data from PJM: %s", err)
            return None
        except Exception as err:
            _LOGGER.error("Unexpected error fetching LMP 5-min data: %s", err)
            return None

    def _get_time_range(self, minutes=10):
        """Generate time range string for API queries."""
        end_time_utc = dt.now(timezone.utc)
        start_time_utc = end_time_utc - timedelta(minutes=minutes)
        return f"{start_time_utc.strftime('%m/%e/%Y %H:%M')}to{end_time_utc.strftime('%m/%e/%Y %H:%M')}"

    def _format_time_range(self, start, end):
        """Format time range for API queries."""
        return f"{start.strftime('%m/%e/%Y %H:%M')}to{end.strftime('%m/%e/%Y %H:%M')}"
