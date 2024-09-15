"""Constants for HA PJM Sensor integration."""
DOMAIN = "ha_pjm_sensor"

CONF_API_KEY = "api_key"
CONF_UPDATE_FREQUENCY = "update_frequency"
CONF_MONITORED_VARIABLES = "monitored_variables"

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
