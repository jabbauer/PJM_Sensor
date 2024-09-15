"""Config flow for HA PJM Sensor integration."""
import logging
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback

from .const import DOMAIN, ZONE_TO_PNODE_ID

_LOGGER = logging.getLogger(__name__)

@config_entries.HANDLERS.register(DOMAIN)
class PJMSensorConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PJM Sensor."""

    VERSION = 1
    CONNECTION_CLASS = config_entries.CONN_CLASS_CLOUD_POLL

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        if user_input is not None:
            # Validate inputs if necessary
            return await self.async_step_options()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Optional("api_key", default=""): str,
            }),
            description_placeholders={"title": "PJM Sensor API Key"},
        )

    async def async_step_options(self, user_input=None):
        """Handle options step: selecting zones, sensors, and update frequencies."""
        if user_input is not None:
            # Store the configuration
            return self.async_create_entry(title="PJM Sensor", data=user_input)

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
            vol.Required("api_key", default=""): str,
            vol.Required("zone"): vol.In(available_zones),
            vol.Required("update_frequency", default=300): vol.All(vol.Coerce(int), vol.Range(min=60, max=86400)),
            **sensor_options
        })

        return self.async_show_form(
            step_id="options",
            data_schema=data_schema,
            description_placeholders={"title": "PJM Sensor Configuration"},
        )
