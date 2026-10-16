"""Minimal client for the Huawei HiLink web API (B315, B310, B525, E5186 and similar LTE CPEs).

These routers expose no SSH and cannot export CSI; the API is used only to identify the
router, read its Wi-Fi settings and check that it is alive. Endpoints used here are
readable without logging in, using the anonymous session token from SesTokInfo.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Dict

import httpx

logger = logging.getLogger(__name__)

# HiLink error codes seen on anonymous requests
LOGIN_REQUIRED_CODES = {"100003", "125002"}


class HiLinkError(Exception):
    """Raised when the router returns an <error> response or is unreachable."""

    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


class HuaweiHiLinkClient:
    """Async read-only client for a Huawei HiLink router."""

    def __init__(self, host: str, timeout: float = 5.0, transport: httpx.AsyncBaseTransport = None):
        self.base_url = f"http://{host}"
        self.timeout = timeout
        self._transport = transport

    async def _get(self, path: str) -> Dict[str, str]:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                         transport=self._transport) as client:
                token_resp = await client.get("/api/webserver/SesTokInfo")
                tokens = _parse_xml(token_resp.text)
                headers = {
                    "Cookie": tokens.get("SesInfo", ""),
                    "__RequestVerificationToken": tokens.get("TokInfo", ""),
                }
                resp = await client.get(f"/api/{path}", headers=headers)
                return _parse_xml(resp.text)
        except httpx.HTTPError as e:
            raise HiLinkError(f"router at {self.base_url} unreachable: {e}") from e

    async def get_basic_information(self) -> Dict[str, str]:
        """Model name (devicename), firmware and product family."""
        return await self._get("device/basic_information")

    async def get_wlan_settings(self) -> Dict[str, str]:
        """SSID, channel (0 = auto), mode (b/g/n), bandwidth, antenna count."""
        return await self._get("wlan/basic-settings")

    async def get_status(self) -> Dict[str, str]:
        """Connection status, Wi-Fi state and number of Wi-Fi clients."""
        return await self._get("monitoring/status")

    async def is_alive(self) -> bool:
        try:
            await self.get_status()
            return True
        except HiLinkError as e:
            # The web server answered, it just wants a login: the router is up.
            return e.code in LOGIN_REQUIRED_CODES


def _parse_xml(text: str) -> Dict[str, str]:
    """Flatten a HiLink <response> into a dict; raise HiLinkError for <error>."""
    try:
        root = ET.fromstring(text.strip())
    except ET.ParseError as e:
        raise HiLinkError(f"invalid XML from router: {e}") from e

    if root.tag == "error":
        code = root.findtext("code", default="")
        raise HiLinkError(f"router returned error {code}", code=code)
    return {child.tag: (child.text or "") for child in root}
