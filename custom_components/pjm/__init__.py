"""The PJM integration."""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.const import Platform

from .const import (
    CONF_COVER_MODE,
    CONF_EXPERIMENTAL_COINCIDENT,
    DEFAULT_COVER_MODE,
)

_LOGGER = logging.getLogger(__name__)

DOMAIN = "pjm"
PLATFORMS = [Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up PJM from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_options))
    return True


async def _async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Rebuild sensors after saved configuration changes."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload PJM config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry when ConfigFlow.VERSION increases.

    v1 → v2 (2.3.0): add cover_mode + experimental_coincident_logic defaults.
    Existing sensors / zone / thresholds / API key are preserved.
    """
    _LOGGER.info(
        "Migrating PJM config entry %s from version %s",
        entry.entry_id,
        entry.version,
    )

    if entry.version == 1:
        new_data = {**entry.data}
        new_data.setdefault(CONF_COVER_MODE, DEFAULT_COVER_MODE)
        new_data.setdefault(CONF_EXPERIMENTAL_COINCIDENT, False)
        hass.config_entries.async_update_entry(entry, data=new_data, version=2)
        _LOGGER.info(
            "Migrated PJM entry %s to version 2 (cover_mode=%s, experimental=%s)",
            entry.entry_id,
            new_data.get(CONF_COVER_MODE),
            new_data.get(CONF_EXPERIMENTAL_COINCIDENT),
        )

    if entry.version > 2:
        _LOGGER.error(
            "PJM config entry %s is version %s; this integration only supports up to 2",
            entry.entry_id,
            entry.version,
        )
        return False

    return True
