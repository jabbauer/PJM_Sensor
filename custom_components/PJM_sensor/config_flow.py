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

# Default sensors: Only the first 3 are enabled by default
DEFAULT_SENSORS = {"instantaneous_total_load", "total_short_forecast", "total_load_forecast"}

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
            # If no sensors are selected, show an error.
            if not selected_sensors:
                errors["base"] = "no_sensors_selected"
                return self.async_show_form(
                    step_id="user",
                    data_schema=self._build_schema(user_input),
                    errors=errors
                )
            # If no API key is provided, limit the sensors to a maximum of 3.
            if not user_input.get("api_key") and len(selected_sensors) > 3:
                errors["base"] = "max_sensors_exceeded_without_api_key"
                return self.async_show_form(
                    step_id="user",
                    data_schema=self._build_schema(user_input),
                    errors=errors
                )

            self.entry_data = {
                "zone": user_input["zone"],
                "api_key": user_input.get("api_key"),  # Now optional!
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
        """Build the configuration schema with our custom defaults."""
        defaults = defaults or {
            "zone": "PJM-RTO",
            "api_key": "",  # API key is now optional
            "peak_threshold": DEFAULT_PEAK_THRESHOLD,
            "accuracy_threshold": DEFAULT_ACCURACY_THRESHOLD
        }
        schema_dict = {
            vol.Required("zone", default=defaults.get("zone")): vol.In(self.zone_list),
            vol.Optional("api_key", default=defaults.get("api_key")): str,  # No longer required!
            vol.Optional("peak_threshold", default=defaults.get("peak_threshold")): cv.positive_int,
            vol.Optional("accuracy_threshold", default=defaults.get("accuracy_threshold")): vol.Coerce(float),
        }
        # Default only the first three sensors as selected
        for sensor_id, sensor_label in SENSOR_OPTIONS.items():
            default_value = sensor_id in DEFAULT_SENSORS
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
            # Enforce the sensor limit if no API key is provided.
            if not user_input.get("api_key") and len(selected_sensors) > 3:
                errors["base"] = "max_sensors_exceeded_without_api_key"
                return self.async_show_form(
                    step_id="init",
                    data_schema=self._build_schema(user_input),
                    errors=errors
                )
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
                "api_key": user_input.get("api_key"),  # API key remains optional
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
        """Build the options flow schema with our custom defaults."""
        schema_dict = {
            vol.Required("zone", default=data.get("zone")): vol.In(sorted(ZONE_TO_PNODE_ID.keys())),
            vol.Optional("api_key", default=data.get("api_key")): str,  # API key stays optional
            vol.Optional("peak_threshold", default=data.get("peak_threshold", DEFAULT_PEAK_THRESHOLD)): cv.positive_int,
            vol.Optional("accuracy_threshold", default=data.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD)): vol.Coerce(float),
        }
        # Default only the first three sensors as selected
        for sensor_id, sensor_label in SENSOR_OPTIONS.items():
            default_value = sensor_id in DEFAULT_SENSORS
            schema_dict[vol.Optional(sensor_id, default=default_value)] = bool
        return vol.Schema(schema_dict)
