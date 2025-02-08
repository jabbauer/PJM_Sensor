"""Constants for HA PJM Sensor integration."""
DOMAIN = "pjm"

CONF_API_KEY = "api_key"
CONF_UPDATE_FREQUENCY = "update_frequency"
CONF_MONITORED_VARIABLES = "monitored_variables"

# Sensor Types
CONF_INSTANTANEOUS_ZONE_LOAD = 'instantaneous_zone_load'
CONF_INSTANTANEOUS_TOTAL_LOAD = 'instantaneous_total_load'
CONF_ZONE_LOAD_FORECAST = 'zone_load_forecast'
CONF_TOTAL_LOAD_FORECAST = 'total_load_forecast'
CONF_ZONE_SHORT_FORECAST = 'zone_short_forecast'
CONF_TOTAL_SHORT_FORECAST = 'total_short_forecast'
CONF_ZONAL_LMP = 'zonal_lmp'
# NEW: Coincident Peak Prediction Sensor
CONF_COINCIDENT_PEAK_PREDICTION = "coincident_peak_prediction"

# New configuration options for the peak predictor
CONF_PEAK_THRESHOLD = "peak_threshold"
CONF_ACCURACY_THRESHOLD = "accuracy_threshold"

# Defaults for new configuration options
DEFAULT_PEAK_THRESHOLD = 140000  # MW
DEFAULT_ACCURACY_THRESHOLD = 0.8

# Define available zones
AVAILABLE_ZONES = [
    'PJM-RTO', 'MID-ATL/APS', 'AECO', 'BGE', 'DPL', 'JCPL', 'METED',
    'PECO', 'PEPCO', 'PPL', 'PENELEC', 'PSEG', 'RECO', 'APS', 'AEP',
    'COMED', 'DAY', 'DOM', 'DUQ', 'ATSI', 'DEOK', 'EKPC', 'OVEC'
]

# Define zone to PNode ID mapping
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
    'OVEC': 1709725933
}

# Define sensor types and their units
SENSOR_TYPES = {
    CONF_INSTANTANEOUS_ZONE_LOAD: ["Zone Load", 'MW'],
    CONF_INSTANTANEOUS_TOTAL_LOAD: ["PJM Total Load", 'MW'],
    CONF_ZONE_LOAD_FORECAST: ["Load Forecast", 'MW'],
    CONF_TOTAL_LOAD_FORECAST: ["PJM Total Load Forecast", 'MW'],
    CONF_ZONE_SHORT_FORECAST: ["Zone 2HR Forecast", "MW"],
    CONF_TOTAL_SHORT_FORECAST: ["PJM 2HR Forecast", "MW"],
    CONF_ZONAL_LMP: ["Hourly Average Zonal LMP", '$/MWh'],
    CONF_COINCIDENT_PEAK_PREDICTION: ["Coincident Peak Prediction", "MW"],
}
