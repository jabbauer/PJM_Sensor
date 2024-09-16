"""Config flow for HA PJM Sensor integration."""
import logging
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback

from .const import DOMAIN, ZONE_TO_PNODE_ID, SENSOR_TYPES

_LOGGER = logging.getLogger(__name__)

@config_entries.HANDLERS.register(DOMAIN)
class PJMSensorConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PJM Sensor."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_CLOUD_POLL

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        if user_input is not None:
            # Proceed to configuration step
            return await self.async_step_config(user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("api_key", default=""): vol.Coerce(str),
            }),
            description_placeholders={"title": "PJM Sensor API Key"},
        )

    async def async_step_config(self, user_input=None):
        """Handle configuration step: zone, update frequency, sensor options."""
        if user_input is not None:
            # Extract configuration
            api_key = user_input.get("api_key")
            zone = user_input.get("zone")
            update_frequency = user_input.get("update_frequency")

            # Collect selected sensors
            monitored_variables = []
            sensor_mapping = {
                "instantaneous_zone_load": "instantaneous_zone_load",
                "instantaneous_total_load": "instantaneous_total_load",
                "zone_load_forecast": "zone_load_forecast",
                "total_load_forecast": "total_load_forecast",
                "zone_short_forecast": "zone_short_forecast",
                "total_short_forecast": "total_short_forecast",
                "zonal_lmp_avg": "zonal_lmp_avg",
                "zonal_lmp_5min": "zonal_lmp_5min",
            }

            for key, sensor_type in sensor_mapping.items():
                if user_input.get(key):
                    monitored_variables.append({
                        "type": sensor_type,
                        "zone": zone,
                        "name": f"{zone} {SENSOR_TYPES[sensor_type][0]}"
                    })

            if not monitored_variables:
                _LOGGER.error("No sensors selected to monitor.")
                return self.async_show_form(
                    step_id="config",
                    data_schema=self._build_config_schema(),
                    errors={"base": "no_sensors_selected"},
                )

            return self.async_create_entry(
                title="PJM Sensor",
                data={
                    "api_key": api_key,
                    "zone": zone,
                    "update_frequency": update_frequency,
                    "monitored_variables": monitored_variables,
                },
            )

        # Define available zones
        available_zones = list(ZONE_TO_PNODE_ID.keys())

        # Define sensor options
        sensor_options = {
            vol.Optional("instantaneous_zone_load", default=True): bool,
            vol.Optional("instantaneous_total_load", default=True): bool,
            vol.Optional("zone_load_forecast", default=False): bool,
            vol.Optional("total_load_forecast", default=False): bool,
            vol.Optional("zone_short_forecast", default=False): bool,
            vol.Optional("total_short_forecast", default=False): bool,
            vol.Optional("zonal_lmp_avg", default=False): bool,
            vol.Optional("zonal_lmp_5min", default=False): bool,
        }

        data_schema = vol.Schema({
            vol.Required("zone"): vol.In(available_zones),
            vol.Required("update_frequency", default=300): vol.All(vol.Coerce(int), vol.Range(min=60, max=86400)),
            **sensor_options
        })

        return self.async_show_form(
            step_id="config",
            data_schema=data_schema,
            description_placeholders={"title": "PJM Sensor Configuration"},
        )

    def _build_config_schema(self):
        """Build config schema for re-showing the config form."""
        available_zones = list(ZONE_TO_PNODE_ID.keys())

        sensor_options = {
            vol.Optional("instantaneous_zone_load", default=True): bool,
            vol.Optional("instantaneous_total_load", default=True): bool,
            vol.Optional("zone_load_forecast", default=False): bool,
            vol.Optional("total_load_forecast", default=False): bool,
            vol.Optional("zone_short_forecast", default=False): bool,
            vol.Optional("total_short_forecast", default=False): bool,
            vol.Optional("zonal_lmp_avg", default=False): bool,
            vol.Optional("zonal_lmp_5min", default=False): bool,
        }

        return vol.Schema({
            vol.Required("zone"): vol.In(available_zones),
            vol.Required("update_frequency", default=300): vol.All(vol.Coerce(int), vol.Range(min=60, max=86400)),
            **sensor_options
        })
