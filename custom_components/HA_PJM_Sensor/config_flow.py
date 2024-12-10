"""Config flow for PJM Sensor integration."""
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
import homeassistant.helpers.config_validation as cv

from . import DOMAIN

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

SENSOR_OPTIONS = {
    "instantaneous_zone_load": "Zonal Instantaneous Load",
    "instantaneous_total_load": "System Instantaneous Load",
    "zone_load_forecast": "Zonal Daily Forecast",
    "total_load_forecast": "System Daily Forecast",
    "zone_short_forecast": "Zonal 2Hr Forecast",
    "total_short_forecast": "System 2Hr Forecast",
    "zonal_lmp": "Zonal LMP"
}

class PJMConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PJM integration."""

    VERSION = 1
    entry_data = {}
    zone_list = sorted(ZONE_TO_PNODE_ID.keys())

    async def async_step_user(self, user_input=None) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            # Store selected zone and move to the next step
            self.entry_data["zone"] = user_input["zone"]
            return await self.async_step_sensors()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("zone", default="PJM-RTO"): vol.In(self.zone_list),
            }),
            errors=errors
        )

    async def async_step_sensors(self, user_input=None) -> FlowResult:
        """Select which sensors to enable."""
        errors = {}

        if user_input is not None:
            self.entry_data["sensors"] = user_input["sensors"]
            return self.async_create_entry(title="PJM Integration", data=self.entry_data)

        return self.async_show_form(
            step_id="sensors",
            data_schema=vol.Schema({
                vol.Required("sensors", default=list(SENSOR_OPTIONS.keys())): cv.multi_select(SENSOR_OPTIONS)
            }),
            errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return PJMOptionsFlowHandler(config_entry)


class PJMOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for PJM."""

    def __init__(self, config_entry):
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the PJM options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        sensors = self.config_entry.data.get("sensors", [])
        zone = self.config_entry.data.get("zone", "PJM-RTO")
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required("zone", default=zone): vol.In(sorted(ZONE_TO_PNODE_ID.keys())),
                vol.Required("sensors", default=sensors): cv.multi_select(SENSOR_OPTIONS)
            })
        )
