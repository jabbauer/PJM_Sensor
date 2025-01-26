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
        """Handle the configuration in a single step."""
        errors = {}

        if user_input is not None:
            # Collect selected sensors
            selected_sensors = [
                sensor_id 
                for sensor_id in SENSOR_OPTIONS 
                if user_input.get(sensor_id, False)
            ]
            
            if not selected_sensors:
                errors["base"] = "no_sensors_selected"
                return self.async_show_form(
                    step_id="user",
                    data_schema=self._build_schema(user_input),
                    errors=errors
                )
            
            # Save configuration
            self.entry_data = {
                "zone": user_input["zone"],
                "sensors": selected_sensors
            }
            return self.async_create_entry(title="PJM Integration", data=self.entry_data)

        # Build schema dynamically
        return self.async_show_form(
            step_id="user",
            data_schema=self._build_schema(),
            errors=errors
        )

    def _build_schema(self, defaults=None):
        """Build schema with zone dropdown and individual sensor checkboxes."""
        defaults = defaults or {"zone": "PJM-RTO"}
        schema = {
            vol.Required("zone", default=defaults.get("zone")): vol.In(self.zone_list)
        }
        
        # Add a checkbox for each sensor (default all checked)
        for sensor_id, sensor_label in SENSOR_OPTIONS.items():
            schema[
                vol.Optional(
                    sensor_id,
                    default=defaults.get(sensor_id, True)  # Default all sensors selected
                )
            ] = bool
        
        return vol.Schema(schema)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return PJMOptionsFlowHandler(config_entry)


class PJMOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for PJM."""

    def __init__(self, config_entry):
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage PJM options with checkboxes."""
        errors = {}
        current_sensors = self.config_entry.data.get("sensors", [])
        current_zone = self.config_entry.data.get("zone", "PJM-RTO")

        if user_input is not None:
            selected_sensors = [
                sensor_id 
                for sensor_id in SENSOR_OPTIONS 
                if user_input.get(sensor_id, False)
            ]
            
            if not selected_sensors:
                errors["base"] = "no_sensors_selected"
                return self.async_show_form(
                    step_id="init",
                    data_schema=self._build_schema(user_input),
                    errors=errors
                )
            
            # Update configuration
            updated_data = {**self.config_entry.data}
            updated_data.update({
                "zone": user_input["zone"],
                "sensors": selected_sensors
            })
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data=updated_data
            )
            return self.async_create_entry(title="", data={})

        # Build schema with current selections
        return self.async_show_form(
            step_id="init",
            data_schema=self._build_schema(current_zone, current_sensors),
            errors=errors
        )

    def _build_schema(self, zone, sensors=None):
        """Build schema with checkboxes reflecting current selections."""
        sensors = sensors or []
        schema = {
            vol.Required("zone", default=zone): vol.In(sorted(ZONE_TO_PNODE_ID.keys()))
        }
        
        for sensor_id, sensor_label in SENSOR_OPTIONS.items():
            schema[
                vol.Optional(
                    sensor_id,
                    default=sensor_id in sensors  # Preserve existing selections
                )
            ] = bool
        
        return vol.Schema(schema)
