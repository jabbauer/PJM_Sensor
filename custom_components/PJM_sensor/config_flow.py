"""Config flow for PJM Sensor integration."""
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    ZONE_TO_PNODE_ID,
    DEFAULT_PEAK_THRESHOLD,
    DEFAULT_ACCURACY_THRESHOLD,
    CONF_API_KEY,
)

SENSOR_OPTIONS = {
    "instantaneous_total_load": "System Instantaneous Load",
    "total_short_forecast": "System 2Hr Forecast",
    "total_load_forecast": "System Daily Forecast",
    "instantaneous_zone_load": "Zonal Instantaneous Load",
    "zone_short_forecast": "Zonal 2Hr Forecast",
    "zone_load_forecast": "Zonal Daily Forecast",
    "zonal_lmp": "Zonal LMP",
    "coincident_peak_prediction_zone": "Coincident Peak Prediction (Zone)",
    "coincident_peak_prediction_system": "Coincident Peak Prediction (System)",
}

SYSTEM_SENSORS = {
    "instantaneous_total_load",
    "total_short_forecast",
    "total_load_forecast",
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
            self.entry_data = {
                "zone": user_input["zone"],
                "api_key": user_input["api_key"],
                "sensors": selected_sensors,
                "peak_threshold": user_input.get("peak_threshold", DEFAULT_PEAK_THRESHOLD),
                "accuracy_threshold": user_input.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD),
            }
            return self.async_create_entry(title="PJM Integration", data=self.entry_data)

        return self.async_show_form(
            step_id="user",
            data_schema=self._build_schema(),
            errors=errors
        )

    def _build_schema(self, defaults=None):
        defaults = defaults or {
            "zone": "PJM-RTO",
            "api_key": "",
            "peak_threshold": DEFAULT_PEAK_THRESHOLD,
            "accuracy_threshold": DEFAULT_ACCURACY_THRESHOLD
        }
        schema_dict = {
            vol.Required("zone", default=defaults.get("zone")): vol.In(self.zone_list),
            vol.Required("api_key", default=defaults.get("api_key")): str,
            vol.Optional("peak_threshold", default=defaults.get("peak_threshold")): cv.positive_int,
            vol.Optional("accuracy_threshold", default=defaults.get("accuracy_threshold")): vol.Coerce(float),
        }
        for sensor_id, sensor_label in SENSOR_OPTIONS.items():
            default_value = defaults.get(sensor_id, sensor_id in SYSTEM_SENSORS or sensor_id.startswith("coincident_peak_prediction"))
            schema_dict[vol.Optional(sensor_id, default=default_value)] = bool
        return vol.Schema(schema_dict)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return PJMOptionsFlowHandler(config_entry)


class PJMOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for PJM."""
    def __init__(self, config_entry):
        self._config_entry = config_entry

    async def async_step_init(self, user_input=None):
        errors = {}
        current_sensors = self._config_entry.data.get("sensors", [])
        current_zone = self._config_entry.data.get("zone", "PJM-RTO")
        current_api_key = self._config_entry.data.get("api_key", "")
        current_peak_threshold = self._config_entry.data.get("peak_threshold", DEFAULT_PEAK_THRESHOLD)
        current_accuracy_threshold = self._config_entry.data.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD)

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
            updated_data = {**self._config_entry.data}
            updated_data.update({
                "zone": user_input["zone"],
                "api_key": user_input["api_key"],
                "sensors": selected_sensors,
                "peak_threshold": user_input.get("peak_threshold", DEFAULT_PEAK_THRESHOLD),
                "accuracy_threshold": user_input.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD),
            })
            self.hass.config_entries.async_update_entry(
                self._config_entry,
                data=updated_data
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="init",
            data_schema=self._build_schema({
                "zone": current_zone,
                "api_key": current_api_key,
                "peak_threshold": current_peak_threshold,
                "accuracy_threshold": current_accuracy_threshold,
            }),
            errors=errors
        )

    def _build_schema(self, data):
        schema_dict = {
            vol.Required("zone", default=data.get("zone")): vol.In(sorted(ZONE_TO_PNODE_ID.keys())),
            vol.Required("api_key", default=data.get("api_key")): str,
            vol.Optional("peak_threshold", default=data.get("peak_threshold", DEFAULT_PEAK_THRESHOLD)): cv.positive_int,
            vol.Optional("accuracy_threshold", default=data.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD)): vol.Coerce(float),
        }
        for sensor_id, sensor_label in SENSOR_OPTIONS.items():
            schema_dict[vol.Optional(sensor_id, default=data.get(sensor_id, sensor_id in SYSTEM_SENSORS or sensor_id.startswith("coincident_peak_prediction")))] = bool
        return vol.Schema(schema_dict)
