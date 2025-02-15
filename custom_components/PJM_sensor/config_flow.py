"""Config flow for PJM Sensor integration."""
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    ZONE_TO_PNODE_ID,
    DEFAULT_ACCURACY_THRESHOLD,
    CONF_API_KEY,
    CONF_COINCIDENT_PEAK_PREDICTION_ZONE,
    CONF_COINCIDENT_PEAK_PREDICTION_SYSTEM,
)

SENSOR_OPTIONS = {
    "instantaneous_total_load": "System Instantaneous Load",
    "total_short_forecast": "System 2Hr Forecast",
    "total_load_forecast": "System Daily Forecast",
    "instantaneous_zone_load": "Zonal Instantaneous Load",
    "zone_short_forecast": "Zonal 2Hr Forecast",
    "zone_load_forecast": "Zonal Daily Forecast",
    "zonal_lmp": "Zonal LMP",
    CONF_COINCIDENT_PEAK_PREDICTION_ZONE: "Coincident Peak Prediction (Zone)",
    CONF_COINCIDENT_PEAK_PREDICTION_SYSTEM: "Coincident Peak Prediction (System)",
}

# Default sensors: Only the first 3 are enabled by default.
DEFAULT_SENSORS = {"instantaneous_total_load", "total_short_forecast", "total_load_forecast"}

# New default thresholds for coincident peak predictions.
DEFAULT_PEAK_THRESHOLD_ZONE = 16500
DEFAULT_PEAK_THRESHOLD_SYSTEM = 140000

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
                sensor_id for sensor_id in SENSOR_OPTIONS if user_input.get(sensor_id, False)
            ]
            if not selected_sensors:
                errors["base"] = "no_sensors_selected"
                return self.async_show_form(
                    step_id="user",
                    data_schema=self._build_schema(user_input),
                    errors=errors,
                )
            # Without an API key, limit to a maximum of 3 sensors.
            if not user_input.get("api_key") and len(selected_sensors) > 3:
                errors["base"] = "max_sensors_exceeded_without_api_key"
                return self.async_show_form(
                    step_id="user",
                    data_schema=self._build_schema(user_input),
                    errors=errors,
                )

            self.entry_data = {
                "zone": user_input["zone"],
                "api_key": user_input.get("api_key"),
                "sensors": selected_sensors,
                "peak_threshold_zone": user_input.get("peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE),
                "peak_threshold_system": user_input.get("peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM),
                "accuracy_threshold": user_input.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD),
            }
            return self.async_create_entry(title="PJM Integration", data=self.entry_data)

        return self.async_show_form(
            step_id="user",
            data_schema=self._build_schema(),
            errors=errors,
        )

    def _build_schema(self, defaults=None):
        """Build the configuration schema.
        
        Order: zone, api_key, sensor checkboxes, then peak threshold fields, then accuracy_threshold.
        """
        defaults = defaults or {
            "zone": "PJM-RTO",
            "api_key": "",
            "peak_threshold_zone": DEFAULT_PEAK_THRESHOLD_ZONE,
            "peak_threshold_system": DEFAULT_PEAK_THRESHOLD_SYSTEM,
            "accuracy_threshold": DEFAULT_ACCURACY_THRESHOLD,
        }
        schema_dict = {}
        # General settings.
        schema_dict[vol.Required("zone", default=defaults.get("zone"))] = vol.In(self.zone_list)
        schema_dict[vol.Optional("api_key", default=defaults.get("api_key"))] = str
        # Sensor selection checkboxes.
        for sensor_id, sensor_label in SENSOR_OPTIONS.items():
            default_value = sensor_id in DEFAULT_SENSORS
            schema_dict[vol.Optional(sensor_id, default=default_value)] = bool
        # Peak threshold fields (added after sensor checkboxes).
        schema_dict[vol.Optional("peak_threshold_zone", default=defaults.get("peak_threshold_zone"))] = cv.positive_int
        schema_dict[vol.Optional("peak_threshold_system", default=defaults.get("peak_threshold_system"))] = cv.positive_int
        # Accuracy threshold.
        schema_dict[vol.Optional("accuracy_threshold", default=defaults.get("accuracy_threshold"))] = vol.Coerce(float)
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
        current_data = self._config_entry.data
        if user_input is not None:
            selected_sensors = [
                sensor_id for sensor_id in SENSOR_OPTIONS if user_input.get(sensor_id, False)
            ]
            if not user_input.get("api_key") and len(selected_sensors) > 3:
                errors["base"] = "max_sensors_exceeded_without_api_key"
                return self.async_show_form(
                    step_id="init",
                    data_schema=self._build_schema(user_input),
                    errors=errors,
                )
            if not selected_sensors:
                errors["base"] = "no_sensors_selected"
                return self.async_show_form(
                    step_id="init",
                    data_schema=self._build_schema(user_input),
                    errors=errors,
                )
            updated_data = {**current_data}
            updated_data.update({
                "zone": user_input["zone"],
                "api_key": user_input.get("api_key"),
                "sensors": selected_sensors,
                "peak_threshold_zone": user_input.get("peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE),
                "peak_threshold_system": user_input.get("peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM),
                "accuracy_threshold": user_input.get("accuracy_threshold", current_data.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD)),
            })
            self.hass.config_entries.async_update_entry(self._config_entry, data=updated_data)
            return self.async_create_entry(title="", data={})
        return self.async_show_form(
            step_id="init",
            data_schema=self._build_schema({
                "zone": current_data.get("zone", "PJM-RTO"),
                "api_key": current_data.get("api_key", ""),
                "peak_threshold_zone": current_data.get("peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE),
                "peak_threshold_system": current_data.get("peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM),
                "accuracy_threshold": current_data.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD),
            }),
            errors=errors,
        )

    def _build_schema(self, data):
        schema_dict = {}
        # General settings.
        schema_dict[vol.Required("zone", default=data.get("zone"))] = vol.In(sorted(ZONE_TO_PNODE_ID.keys()))
        schema_dict[vol.Optional("api_key", default=data.get("api_key"))] = str
        # Sensor selection checkboxes.
        for sensor_id, sensor_label in SENSOR_OPTIONS.items():
            default_value = sensor_id in DEFAULT_SENSORS
            schema_dict[vol.Optional(sensor_id, default=default_value)] = bool
        # Peak threshold fields.
        schema_dict[vol.Optional("peak_threshold_zone", default=data.get("peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE))] = cv.positive_int
        schema_dict[vol.Optional("peak_threshold_system", default=data.get("peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM))] = cv.positive_int
        # Accuracy threshold.
        schema_dict[vol.Optional("accuracy_threshold", default=data.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD))] = vol.Coerce(float)
        return vol.Schema(schema_dict)
