"""Cloud configuration — environment-variable based.

The SDK reads cloud credentials from env vars at agent startup. If both
KOVIO_API_URL and KOVIO_API_KEY are present (and valid), the agent uses
CloudCampaignStore + CloudEventSink. Otherwise, it falls back to local
behavior (default creative file, local-only event log).

Supports an optional .env file in the current working directory for dev
convenience. The .env file is plain KEY=VALUE lines — no dotenv dependency.

This module is the single place the SDK reads cloud env vars; everything else
goes through `load_cloud_config()`.
"""
from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("kovio.config")


@dataclass(frozen=True)
class CloudConfig:
    """Resolved cloud configuration.

    Attributes are None when the corresponding env var is unset. Use
    `is_configured` to check whether the SDK has enough info to run in
    cloud mode.
    """
    api_url: Optional[str]
    api_key: Optional[str]
    robot_id: str
    api_timeout_seconds: float
    campaign_ttl_seconds: float

    @property
    def is_configured(self) -> bool:
        """True if the SDK can talk to the cloud (both URL and key set)."""
        return bool(self.api_url and self.api_key)

    @property
    def api_key_redacted(self) -> str:
        """Return the key with the secret portion masked for logging."""
        if not self.api_key:
            return "(unset)"
        if len(self.api_key) <= 12:
            return "kov_live_••••"
        return f"{self.api_key[:12]}••••"


def _maybe_load_dotenv(cwd: Path | None = None) -> None:
    """Load .env from CWD into os.environ if it exists.

    Existing env vars take precedence (we don't override the shell).
    Comments (lines starting with #), blank lines, and quoted values are
    handled. Multi-line values are not supported.
    """
    cwd = cwd or Path.cwd()
    env_path = cwd / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            # Shell env wins
            if key and key not in os.environ:
                os.environ[key] = value
        log.debug("Loaded .env from %s", env_path)
    except OSError as e:
        log.warning("Could not read %s: %s", env_path, e)


def load_cloud_config(cwd: Path | None = None) -> CloudConfig:
    """Resolve cloud config from env vars (and optional .env file).

    Returns a CloudConfig — check .is_configured to see if the SDK
    should run in cloud mode.
    """
    _maybe_load_dotenv(cwd)

    api_url = os.environ.get("KOVIO_API_URL", "").strip() or None
    api_key = os.environ.get("KOVIO_API_KEY", "").strip() or None
    robot_id = os.environ.get("KOVIO_ROBOT_ID", "").strip() or socket.gethostname()

    try:
        api_timeout = float(os.environ.get("KOVIO_API_TIMEOUT", "8"))
    except ValueError:
        log.warning("KOVIO_API_TIMEOUT not a valid float; using 8.0")
        api_timeout = 8.0

    try:
        ttl = float(os.environ.get("KOVIO_CAMPAIGN_TTL", "300"))
    except ValueError:
        log.warning("KOVIO_CAMPAIGN_TTL not a valid float; using 300.0")
        ttl = 300.0

    config = CloudConfig(
        api_url=api_url,
        api_key=api_key,
        robot_id=robot_id,
        api_timeout_seconds=api_timeout,
        campaign_ttl_seconds=ttl,
    )
    if config.is_configured:
        log.info("Cloud config: url=%s key=%s robot_id=%s",
                 config.api_url, config.api_key_redacted, config.robot_id)
    else:
        log.info("Cloud config: not configured (local fallback mode)")
    return config
