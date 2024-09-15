"""Constants for HA PJM Sensor integration."""
DOMAIN = "ha_pjm_sensor"

CONF_API_KEY = "api_key"
CONF_UPDATE_FREQUENCY = "update_frequency"
CONF_MONITORED_VARIABLES = "monitored_variables"

# Define available zones (should match ZONE_TO_PNODE_ID in sensor.py)
AVAILABLE_ZONES = [
    'PJM-RTO', 'MID-ATL/APS', 'AECO', 'BGE', 'DPL', 'JCPL', 'METED',
    'PECO', 'PEPCO', 'PPL', 'PENELEC', 'PSEG', 'RECO', 'APS', 'AEP',
    'COMED', 'DAY', 'DOM', 'DUQ', 'ATSI', 'DEOK', 'EKPC', 'OVEC'
]
