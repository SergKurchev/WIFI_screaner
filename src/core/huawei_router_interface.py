"""
Router interface for Huawei HiLink routers (e.g. B315s-22) paired with an ESP32 CSI receiver.

The router transmits; the ESP32 measures CSI on the router's frames and its rows arrive
over UDP. The HiLink API is used for router identification and health checks.
"""

from typing import Any, Dict, Optional

import numpy as np

from src.core.router_interface import RouterInterface
from src.hardware.esp32_csi import NUM_SUBCARRIERS, ESP32UDPCSISource
from src.hardware.huawei_hilink import HiLinkError, HuaweiHiLinkClient


class HuaweiHiLinkRouterInterface(RouterInterface):
    """RouterInterface backed by the HiLink API and an ESP32 UDP CSI source."""

    def __init__(
        self,
        router_id: str,
        host: str,
        transmitter_mac: str,
        udp_host: str = "0.0.0.0",
        udp_port: int = 5500,
        window_size: int = 100,
        mock_mode: bool = False,
        hilink_client: Optional[HuaweiHiLinkClient] = None,
    ):
        super().__init__(router_id=router_id, host=host, port=80, username="", password="",
                         interface="esp32", mock_mode=mock_mode)
        self.transmitter_mac = transmitter_mac
        self.hilink = hilink_client or HuaweiHiLinkClient(host)
        self.csi_source = ESP32UDPCSISource(udp_host, udp_port, transmitter_mac, window_size)
        self.router_info: Dict[str, str] = {}

    async def connect(self):
        if self.mock_mode:
            await super().connect()
            return

        try:
            self.router_info = await self.hilink.get_basic_information()
            self.logger.info(
                f"Router {self.router_id}: {self.router_info.get('devicename', 'unknown model')} "
                f"firmware {self.router_info.get('SoftwareVersion', '?')}"
            )
        except HiLinkError as e:
            # Not fatal: CSI comes from the ESP32, the API is only informational.
            self.logger.warning(f"HiLink API unavailable on {self.host}: {e}")

        await self.csi_source.start()
        self.is_connected = True
        self.error_count = 0

    async def disconnect(self):
        self.csi_source.stop()
        await super().disconnect()

    async def _collect_real_csi_data(self) -> Optional[np.ndarray]:
        return self.csi_source.get_window()

    async def check_health(self) -> bool:
        if not self.is_connected:
            return False
        if self.mock_mode:
            return True
        return await self.hilink.is_alive()

    async def get_status(self) -> Dict[str, Any]:
        status = await super().get_status()
        status["configuration"].update({
            "transmitter_mac": self.transmitter_mac,
            "csi_udp_port": self.csi_source.port,
        })
        status["csi_receiver"] = {
            **self.csi_source.stats,
            "buffered_frames": len(self.csi_source.frames),
            "seconds_since_last_frame": self.csi_source.seconds_since_last_frame(),
        }
        return status

    async def get_router_info(self) -> Dict[str, Any]:
        if self.mock_mode:
            return await super().get_router_info()

        wlan: Dict[str, str] = {}
        try:
            wlan = await self.hilink.get_wlan_settings()
        except HiLinkError as e:
            self.logger.warning(f"Could not read Wi-Fi settings: {e}")

        return {
            "model": self.router_info.get("devicename", "Huawei HiLink"),
            "firmware": self.router_info.get("SoftwareVersion", "Unknown"),
            "wifi_standard": f"802.11{wlan.get('WifiMode', 'b/g/n')}",
            "antennas": int(wlan.get("wifiantennanum") or 2),
            "supported_bands": ["2.4GHz"],
            "wifi_channel": wlan.get("WifiChannel"),  # "0" means auto
            "csi_capabilities": {
                "source": "esp32",
                "max_subcarriers": NUM_SUBCARRIERS,
                "max_antennas": 1,
                "sampling_rate": None,  # set by traffic rate, see esp32_serial_bridge.py
            },
        }

    async def configure_csi_collection(self, config: Dict[str, Any]) -> bool:
        if self.mock_mode:
            return await super().configure_csi_collection(config)
        # Sampling is controlled on the ESP32/bridge side, nothing to configure on the router.
        self.logger.info("CSI collection is configured on the ESP32 bridge, not the router")
        return False
