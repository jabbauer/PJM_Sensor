"""The ha_pjm_sensor integration."""
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the HA PJM Sensor component."""
    # No configuration in configuration.yaml
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Set up HA PJM Sensor from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    # Forward the setup to sensor platform using async_forward_entry_setups
    await hass.config_entries.async_forward_entry_setups(entry, ["sensor"])
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_forward_entry_unload(entry, "sensor")
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok
