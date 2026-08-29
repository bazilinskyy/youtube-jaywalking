"""Common helper functions and configuration utilities for crowd-jaywalking."""

import json
import os
from typing import Any

from custom_logger import CustomLogger

root_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(root_dir, "_cache")
log_dir = os.path.join(root_dir, "_logs")
output_dir = os.path.join(root_dir, "_output")

logger = CustomLogger(__name__)


def get_secrets(
    entry_name: str,
    secret_file_name: str = "secret",
    secret_default_file_name: str = "default.secret",
) -> Any:
    """Opens the secrets file and returns the requested entry.

    Args:
        entry_name: Key name inside the secret JSON dictionary.
        secret_file_name: Primary secrets filename to load.
        secret_default_file_name: Fallback template filename if secret_file_name is missing.

    Returns:
        The value corresponding to entry_name.
    """
    secret_path = os.path.join(root_dir, secret_file_name)
    default_secret_path = os.path.join(root_dir, secret_default_file_name)

    if os.path.isfile(secret_path):
        with open(secret_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif os.path.isfile(default_secret_path):
        with open(default_secret_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        logger.error("No secrets file found at {} or {}", secret_path, default_secret_path)
        return ""

    return data.get(entry_name, "")


def get_configs(
    entry_name: str,
    config_file_name: str = "config",
    config_default_file_name: str = "default.config",
) -> Any:
    """Opens the config file and returns the requested configuration entry.

    Args:
        entry_name: Top-level key inside the config JSON dictionary.
        config_file_name: Custom configuration filename.
        config_default_file_name: Fallback default configuration filename.

    Returns:
        The configuration object or value for entry_name.
    """
    if not check_config(config_file_name, config_default_file_name):
        logger.warning("Config validation indicated discrepancies. Falling back to default.")

    config_path = os.path.join(root_dir, config_file_name)
    default_config_path = os.path.join(root_dir, config_default_file_name)

    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if entry_name in content:
                    return content[entry_name]
        except Exception as e:
            logger.error("Error reading config {}: {}", config_path, e)

    if os.path.isfile(default_config_path):
        with open(default_config_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            return content.get(entry_name, {})

    logger.error("Config file not found: {}", default_config_path)
    return {}


def check_config(
    config_file_name: str = "config",
    config_default_file_name: str = "default.config",
) -> bool:
    """Validates whether custom config has matching required top-level structure as default.config.

    Args:
        config_file_name: Path/name of the user-provided config file.
        config_default_file_name: Path/name of the baseline template config.

    Returns:
        True if the configuration is valid or default exists; False otherwise.
    """
    config_path = os.path.join(root_dir, config_file_name)
    default_config_path = os.path.join(root_dir, config_default_file_name)

    if not os.path.isfile(config_path):
        return os.path.isfile(default_config_path)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(default_config_path, "r", encoding="utf-8") as f:
            default_config = json.load(f)

        missing_keys = [k for k in default_config if k not in config]
        if missing_keys:
            logger.warning(
                "Config {} is missing keys from {}: {}",
                config_file_name,
                config_default_file_name,
                missing_keys,
            )
            return False
        return True
    except json.decoder.JSONDecodeError:
        logger.error("Config file {} is badly formatted JSON.", config_file_name)
        return False
    except Exception as e:
        logger.error("Unexpected error validating config: {}", e)
        return False
