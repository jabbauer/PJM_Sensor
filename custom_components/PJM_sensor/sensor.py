"""
Support for PJM data.
"""
import asyncio
from datetime import datetime as dt, date, time, timezone, timedelta
import logging
import aiohttp
import async_timeout
import urllib.parse
import time as time_module  # Added for rate limiting, time_module to avoid conflict with datetime above

from homeassistant.helpers.event import async_call_later
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import Throttle
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_LOGGER = logging.getLogger(__name__)

RESOURCE_INSTANTANEOUS = 'https://api.pjm.com/api/v1/inst_load'
RESOURCE_FORECAST = 'https://api.pjm.com/api/v1/load_frcstd_7_day'
RESOURCE_SHORT_FORECAST = 'https://api.pjm.com/api/v1/very_short_load_frcst'
RESOURCE_LMP = 'https://api.pjm.com/api/v1/rt_unverified_fivemin_lmps'
RESOURCE_SUBSCRIPTION_KEY = 'https://dataminer2.pjm.com/config/settings.json'

# Default update frequencies
MIN_TIME_BETWEEN_UPDATES_INSTANTANEOUS = timedelta(seconds=300) #5min for load, LMPs
MIN_TIME_BETWEEN_UPDATES_FORECAST = timedelta(seconds=3600) #1hr for forecasts

PJM_RTO_ZONE = "PJM RTO"
FORECAST_COMBINED_ZONE = 'RTO_COMBINED'

CONF_INSTANTANEOUS_ZONE_LOAD = 'instantaneous_zone_load'
CONF_INSTANTANEOUS_TOTAL_LOAD = 'instantaneous_total_load'
CONF_ZONE_LOAD_FORECAST = 'zone_load_forecast'
CONF_TOTAL_LOAD_FORECAST = 'total_load_forecast'
CONF_ZONE_SHORT_FORECAST = 'zone_short_forecast'
CONF_TOTAL_SHORT_FORECAST = 'total_short_forecast'
CONF_ZONAL_LMP = 'zonal_lmp'

SENSOR_TYPES = {
    CONF_INSTANTANEOUS_ZONE_LOAD: [" Zone Load", 'MW'],
    CONF_INSTANTANEOUS_TOTAL_LOAD: ["PJM System Load", 'MW'],
    CONF_ZONE_LOAD_FORECAST: ["Zone Forecast", 'MW'],
    CONF_TOTAL_LOAD_FORECAST: ["PJM System Forecast", 'MW'],
    CONF_ZONE_SHORT_FORECAST: ["Zone 2HR Forecast", "MW"],
    CONF_TOTAL_SHORT_FORECAST: ["PJM 2HR Forecast", "MW"],
    CONF_ZONAL_LMP: ["Zone LMP",'$/MWh'],
}

ZONE_TO_PNODE_ID = {
    'PJM-RTO': 1,
    'MID-ATL/APS': 3,
    'AECO': 51291,
    'BGE': 51292,
    'DPL': 51293,
    'JCPL': 51295,
    'METED': 51296,
    'PECO': 51297,
    'PEPCO': 51298,
    'PPL': 51299,
    'PENELEC': 51300,
    'PSEG': 51301,
    'RECO': 7633629,
    'APS': 8394954,
    'AEP': 8445784,
    'COMED': 33092371,
    'DAY': 34508503,
    'DOM': 34964545,
    'DUQ': 37737283,
    'ATSI': 116013753,
    'DEOK': 124076095,
    'EKPC': 970242670,
    'OVEC': 1709725933,
}

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
            dev.append(PJMSensor(
                pjm_data, sensor_type,
                pnode_id, None))
        else:
            dev.append(PJMSensor(
                pjm_data, sensor_type,
                identifier, None))

    async_add_entities(dev, False)

#    # Schedule updates starting at 10s, then every 10s
#    for index, entity in enumerate(dev):
#        delay = 10 + (index * 10)
#        async_call_later(
#            hass,
#            delay,
#            lambda _, e=entity: hass.async_create_task(e.async_update())
#        )

    # Schedule staggered updates using async tasks
    for index, entity in enumerate(dev):
        delay = 12 + (index * 12)
        hass.async_create_task(
            schedule_delayed_update(entity, delay)
        )

async def schedule_delayed_update(entity, delay):
    """Schedule an update after a delay using async sleep."""
    await asyncio.sleep(delay)
    await entity.async_update()

class PJMSensor(SensorEntity):
    """Implementation of a PJM sensor."""

    def __init__(self, pjm_data, sensor_type, identifier, name):
        super().__init__()  # Important: call the parent class initializer
        self._pjm_data = pjm_data
        self._type = sensor_type
        self._identifier = identifier
        self._unit_of_measurement = SENSOR_TYPES[sensor_type][1]
        #self._unique_id = f"sensor.pjm_{sensor_type}_{identifier}"  # Generate a unique ID
        self._attr_unique_id = f"pjm_{sensor_type}_{identifier}"
        self._state = None
        self._forecast_data = None

        # Default name
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
        """Return the name of the sensor."""
        return self._attr_name
    
    @property
    def unique_id(self):
        """Return a unique ID for the sensor."""
        return self._attr_unique_id

    @property
    def icon(self):
        """Icon to use in the frontend, if any."""
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
            return "mdi:flash"  # Default icon for other sensors

    @property
    def unit_of_measurement(self):
        """Return the unit of measurement of this entity, if any."""
        return self._unit_of_measurement

    @property
    def native_value(self):
        """Return the state of the sensor."""
        return self._state

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
        """Update the sensor value."""
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

class PJMData:
    """Get and parse data from PJM."""

    def __init__(self, websession):
        """Initialize the data object."""
        self._websession = websession
        self._subscription_key = None
        self._request_times = []
        self._lock = asyncio.Lock()

    async def _rate_limit(self):
        """Enforce ≤5 API calls per minute."""
        async with self._lock:
            now = time_module.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
            while len(self._request_times) >= 5:
                oldest = self._request_times[0]
                wait_time = 60 - (now - oldest) + 1
                _LOGGER.debug("Rate limit reached. Waiting %.2f seconds.", wait_time)
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
        """Fetch PJM API subscription key (rate-limited)."""
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
        """Fetch instantaneous load data."""
        end_time_utc = dt.now().astimezone(timezone.utc)
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
        """Fetch load forecast data."""
        midnight_local = dt.combine(date.today(), time())
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
                    forecast_hour_ending = dt.strptime(item['forecast_datetime_ending_utc'], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone(None)
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
        """Fetch short-term load forecast data."""
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
                        dt.strptime(x['forecast_datetime_ending_utc'],'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone(None)
                    )
                )

                if sorted_data:
                    load = int(sorted_data[0]["forecast_load_mw"])
                    forecast_hour_ending = dt.strptime(sorted_data[0]['forecast_datetime_ending_utc'],'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).astimezone(None)
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

        """Fetch hourly average LMP data."""
        now_utc = dt.now().astimezone(timezone.utc)
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

                #Return hourly average of 5-min LMP
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
