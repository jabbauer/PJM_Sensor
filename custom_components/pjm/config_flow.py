"""Config flow for PJM Sensor integration."""
import probatio as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    ZONE_TO_PNODE_ID,
    DEFAULT_ACCURACY_THRESHOLD,
    DEFAULT_PEAK_THRESHOLD_ZONE,
    DEFAULT_PEAK_THRESHOLD_SYSTEM,
    DEFAULT_COVER_MODE,
    COVER_MODE_OPTIONS,
    CONF_API_KEY,
    CONF_COVER_MODE,
    CONF_EXPERIMENTAL_COINCIDENT,
    CONF_COINCIDENT_PEAK_PREDICTION_ZONE,
    CONF_COINCIDENT_PEAK_PREDICTION_SYSTEM,
    CONF_DA_HRL_LMP,
    CONF_MAX_CURTAILMENT_MINUTES,
    DEFAULT_MAX_CURTAILMENT_MINUTES,
    DA_HRL_LMP_ZONES,
)

SENSOR_OPTIONS = {
    "instantaneous_total_load": "System Instantaneous Load",
    "total_short_forecast": "System 2Hr Forecast",
    "total_load_forecast": "System Daily Forecast",
    "instantaneous_zone_load": "Zonal Instantaneous Load",
    "zone_short_forecast": "Zonal 2Hr Forecast",
    "zone_load_forecast": "Zonal Daily Forecast",
    "zonal_lmp": "Zonal LMP (real-time hourly average)",
    CONF_DA_HRL_LMP: "Day Ahead Hourly LMP",
    CONF_COINCIDENT_PEAK_PREDICTION_ZONE: "Coincident Peak Prediction (Zone)",
    CONF_COINCIDENT_PEAK_PREDICTION_SYSTEM: "Coincident Peak Prediction (System)",
}

DEFAULT_SENSORS = {
    "instantaneous_total_load",
    "total_short_forecast",
    "total_load_forecast",
}


class PJMConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PJM integration."""

    VERSION = 2
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
                    errors=errors,
                )
            if CONF_DA_HRL_LMP in selected_sensors and user_input["zone"] not in DA_HRL_LMP_ZONES:
                return self.async_show_form(step_id="user", data_schema=self._build_schema(user_input),
                                            errors={"base": "unsupported_da_zone"})
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
                "peak_threshold_zone": user_input.get(
                    "peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE
                ),
                "peak_threshold_system": user_input.get(
                    "peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM
                ),
                "accuracy_threshold": user_input.get(
                    "accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD
                ),
                CONF_COVER_MODE: user_input.get(CONF_COVER_MODE, DEFAULT_COVER_MODE),
                CONF_EXPERIMENTAL_COINCIDENT: bool(
                    user_input.get(CONF_EXPERIMENTAL_COINCIDENT, False)
                ),
                CONF_MAX_CURTAILMENT_MINUTES: user_input.get(
                    CONF_MAX_CURTAILMENT_MINUTES, DEFAULT_MAX_CURTAILMENT_MINUTES
                ),
            }
            return self.async_create_entry(title="PJM Integration", data=self.entry_data)

        return self.async_show_form(
            step_id="user",
            data_schema=self._build_schema(),
            errors=errors,
        )

    def _build_schema(self, defaults=None):
        defaults = defaults or {
            "zone": "COMED",
            "api_key": "",
            "peak_threshold_zone": DEFAULT_PEAK_THRESHOLD_ZONE,
            "peak_threshold_system": DEFAULT_PEAK_THRESHOLD_SYSTEM,
            "accuracy_threshold": DEFAULT_ACCURACY_THRESHOLD,
            CONF_COVER_MODE: DEFAULT_COVER_MODE,
            CONF_EXPERIMENTAL_COINCIDENT: False,
        }
        schema_dict = {}
        schema_dict[vol.Required("zone", default=defaults.get("zone"))] = vol.In(
            self.zone_list
        )
        schema_dict[vol.Optional("api_key", default=defaults.get("api_key") or "")] = selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
        )
        for sensor_id, _label in SENSOR_OPTIONS.items():
            default_value = sensor_id in DEFAULT_SENSORS
            if defaults and sensor_id in defaults:
                # when re-showing form after error, user_input may have bools
                default_value = bool(defaults.get(sensor_id, default_value))
            schema_dict[vol.Optional(sensor_id, default=default_value)] = bool
        schema_dict[
            vol.Optional(
                "peak_threshold_zone",
                default=defaults.get("peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE),
            )
        ] = cv.positive_int
        schema_dict[
            vol.Optional(
                "peak_threshold_system",
                default=defaults.get(
                    "peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM
                ),
            )
        ] = cv.positive_int
        schema_dict[
            vol.Optional(
                "accuracy_threshold",
                default=defaults.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD),
            )
        ] = vol.Coerce(float)
        schema_dict[
            vol.Optional(
                CONF_COVER_MODE,
                default=defaults.get(CONF_COVER_MODE, DEFAULT_COVER_MODE),
            )
        ] = vol.In(list(COVER_MODE_OPTIONS.keys()))
        schema_dict[
            vol.Optional(
                CONF_EXPERIMENTAL_COINCIDENT,
                default=bool(defaults.get(CONF_EXPERIMENTAL_COINCIDENT, False)),
            )
        ] = bool
        schema_dict[vol.Optional(CONF_MAX_CURTAILMENT_MINUTES,
                                 default=defaults.get(CONF_MAX_CURTAILMENT_MINUTES,
                                                      DEFAULT_MAX_CURTAILMENT_MINUTES))] = vol.All(
            vol.Coerce(int), vol.Range(min=1, max=180)
        )
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
                sensor_id
                for sensor_id in SENSOR_OPTIONS
                if user_input.get(sensor_id, False)
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
            if CONF_DA_HRL_LMP in selected_sensors and user_input["zone"] not in DA_HRL_LMP_ZONES:
                return self.async_show_form(step_id="init", data_schema=self._build_schema(user_input),
                                            errors={"base": "unsupported_da_zone"})
            updated_data = {**current_data}
            updated_data.update(
                {
                    "zone": user_input["zone"],
                    "api_key": user_input.get("api_key"),
                    "sensors": selected_sensors,
                    "peak_threshold_zone": user_input.get(
                        "peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE
                    ),
                    "peak_threshold_system": user_input.get(
                        "peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM
                    ),
                    "accuracy_threshold": user_input.get(
                        "accuracy_threshold",
                        current_data.get(
                            "accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD
                        ),
                    ),
                    CONF_COVER_MODE: user_input.get(
                        CONF_COVER_MODE, DEFAULT_COVER_MODE
                    ),
                    CONF_EXPERIMENTAL_COINCIDENT: bool(
                        user_input.get(CONF_EXPERIMENTAL_COINCIDENT, False)
                    ),
                    CONF_MAX_CURTAILMENT_MINUTES: user_input.get(
                        CONF_MAX_CURTAILMENT_MINUTES, DEFAULT_MAX_CURTAILMENT_MINUTES
                    ),
                }
            )
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=updated_data
            )
            return self.async_create_entry(title="", data={})

        sensors = set(current_data.get("sensors") or [])
        defaults = {
            "zone": current_data.get("zone", "COMED"),
            "api_key": current_data.get("api_key", ""),
            "peak_threshold_zone": current_data.get(
                "peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE
            ),
            "peak_threshold_system": current_data.get(
                "peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM
            ),
            "accuracy_threshold": current_data.get(
                "accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD
            ),
            CONF_COVER_MODE: current_data.get(CONF_COVER_MODE, DEFAULT_COVER_MODE),
            CONF_EXPERIMENTAL_COINCIDENT: bool(
                current_data.get(CONF_EXPERIMENTAL_COINCIDENT, False)
            ),
            CONF_MAX_CURTAILMENT_MINUTES: current_data.get(
                CONF_MAX_CURTAILMENT_MINUTES, DEFAULT_MAX_CURTAILMENT_MINUTES
            ),
        }
        for sid in SENSOR_OPTIONS:
            defaults[sid] = sid in sensors
        return self.async_show_form(
            step_id="init",
            data_schema=self._build_schema(defaults),
            errors=errors,
        )

    def _build_schema(self, data):
        schema_dict = {}
        schema_dict[vol.Required("zone", default=data.get("zone"))] = vol.In(
            sorted(ZONE_TO_PNODE_ID.keys())
        )
        schema_dict[vol.Optional("api_key", default=data.get("api_key") or "")] = selector.TextSelector(
            selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
        )
        for sensor_id, _label in SENSOR_OPTIONS.items():
            default_value = bool(data.get(sensor_id, sensor_id in DEFAULT_SENSORS))
            schema_dict[vol.Optional(sensor_id, default=default_value)] = bool
        schema_dict[
            vol.Optional(
                "peak_threshold_zone",
                default=data.get("peak_threshold_zone", DEFAULT_PEAK_THRESHOLD_ZONE),
            )
        ] = cv.positive_int
        schema_dict[
            vol.Optional(
                "peak_threshold_system",
                default=data.get("peak_threshold_system", DEFAULT_PEAK_THRESHOLD_SYSTEM),
            )
        ] = cv.positive_int
        schema_dict[
            vol.Optional(
                "accuracy_threshold",
                default=data.get("accuracy_threshold", DEFAULT_ACCURACY_THRESHOLD),
            )
        ] = vol.Coerce(float)
        schema_dict[
            vol.Optional(
                CONF_COVER_MODE, default=data.get(CONF_COVER_MODE, DEFAULT_COVER_MODE)
            )
        ] = vol.In(list(COVER_MODE_OPTIONS.keys()))
        schema_dict[
            vol.Optional(
                CONF_EXPERIMENTAL_COINCIDENT,
                default=bool(data.get(CONF_EXPERIMENTAL_COINCIDENT, False)),
            )
        ] = bool
        schema_dict[vol.Optional(CONF_MAX_CURTAILMENT_MINUTES,
                                 default=data.get(CONF_MAX_CURTAILMENT_MINUTES,
                                                  DEFAULT_MAX_CURTAILMENT_MINUTES))] = vol.All(
            vol.Coerce(int), vol.Range(min=1, max=180)
        )
        return vol.Schema(schema_dict)
