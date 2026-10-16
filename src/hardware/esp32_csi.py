"""ESP32 CSI receiver: parses ESP32-CSI-Tool output and buffers frames from one transmitter.

Routers with closed firmware (e.g. Huawei B315s-22) cannot export CSI. Instead the router
acts as the transmitter and an ESP32 running ESP32-CSI-Tool (active_sta) measures CSI on
frames it receives from the router. Each CSV row looks like:

    CSI_DATA,STA,<mac>,<rssi>,<rate>,...,<channel>,<secondary_channel>,<local_timestamp>,...,<len>,[i0 r0 i1 r1 ...]

The bracketed buffer holds signed int8 pairs (imaginary first, then real) per subcarrier.
"""

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Column indices shared by ESP32-CSI-Tool and Espressif's esp-csi examples
MAC_COL = 2
RSSI_COL = 3
CHANNEL_COL = 16
SECONDARY_CHANNEL_COL = 17
LOCAL_TIMESTAMP_COL = 18

# L-LTF of a 20 MHz frame: 64 subcarriers stored as 0..31, -32..-1.
LLTF_SUBCARRIERS = 64
# Data/pilot subcarriers -26..-1 and 1..26 (DC and guard bands are always zero).
USEFUL_LLTF_INDICES = np.r_[38:64, 1:27]
NUM_SUBCARRIERS = len(USEFUL_LLTF_INDICES)  # 52

_MAC_RE = re.compile(r"^([0-9a-f]{2}:){5}[0-9a-f]{2}$")


class ESP32CSIParseError(ValueError):
    """Raised when an ESP32 CSI line cannot be parsed."""


@dataclass
class ESP32CSIFrame:
    """One CSI measurement reported by the ESP32."""

    mac: str
    rssi: int
    channel: int
    local_timestamp_us: int
    csi: np.ndarray  # complex, shape (NUM_SUBCARRIERS,), ordered -26..-1, 1..26


def normalize_mac(mac: str) -> str:
    """Normalize a MAC address to lowercase colon-separated form."""
    return mac.strip().lower().replace("-", ":")


def parse_esp32_csi_line(line: str) -> ESP32CSIFrame:
    """Parse one CSI_DATA row from ESP32-CSI-Tool / esp-csi.

    Raises:
        ESP32CSIParseError: if the row is malformed or not a 20 MHz L-LTF capture.
    """
    line = line.strip()
    if not line.startswith("CSI_DATA"):
        raise ESP32CSIParseError("not a CSI_DATA row")

    open_idx, close_idx = line.find("["), line.rfind("]")
    if open_idx < 0 or close_idx < open_idx:
        raise ESP32CSIParseError("CSI buffer not found")

    fields = line[:open_idx].rstrip(",\" ").split(",")
    if len(fields) <= LOCAL_TIMESTAMP_COL:
        raise ESP32CSIParseError(f"expected at least {LOCAL_TIMESTAMP_COL + 1} fields, got {len(fields)}")

    mac = normalize_mac(fields[MAC_COL])
    if not _MAC_RE.match(mac):
        raise ESP32CSIParseError(f"invalid MAC: {fields[MAC_COL]!r}")

    try:
        rssi = int(fields[RSSI_COL])
        channel = int(fields[CHANNEL_COL])
        secondary_channel = int(fields[SECONDARY_CHANNEL_COL])
        local_timestamp = int(fields[LOCAL_TIMESTAMP_COL])
        raw = np.array(re.split(r"[\s,]+", line[open_idx + 1:close_idx].strip()), dtype=np.int16)
    except ValueError as e:
        raise ESP32CSIParseError(f"bad numeric field: {e}") from e

    if secondary_channel != 0:
        # L-LTF subcarrier layout differs on 40 MHz channels; set the router to 20 MHz.
        raise ESP32CSIParseError("40 MHz capture not supported; set router bandwidth to 20 MHz")
    if raw.size < 2 * LLTF_SUBCARRIERS:
        raise ESP32CSIParseError(f"CSI buffer too short: {raw.size} values")

    pairs = raw[: 2 * LLTF_SUBCARRIERS].astype(np.float32).reshape(LLTF_SUBCARRIERS, 2)
    lltf = pairs[:, 1] + 1j * pairs[:, 0]
    return ESP32CSIFrame(mac, rssi, channel, local_timestamp, lltf[USEFUL_LLTF_INDICES])


class ESP32UDPCSISource(asyncio.DatagramProtocol):
    """Receives CSI_DATA rows over UDP and keeps a sliding window from one transmitter.

    Rows arrive from scripts/esp32_serial_bridge.py, which forwards the ESP32 serial output.
    """

    def __init__(self, host: str, port: int, transmitter_mac: Optional[str], window_size: int = 100):
        self.host = host
        self.port = port
        self.transmitter_mac = normalize_mac(transmitter_mac) if transmitter_mac else None
        self.window_size = window_size
        self.frames: Deque[ESP32CSIFrame] = deque(maxlen=window_size)
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.stats = {"received": 0, "accepted": 0, "parse_errors": 0, "other_transmitters": 0}
        self.last_frame_time: Optional[float] = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: self, local_addr=(self.host, self.port))
        logger.info(f"Listening for ESP32 CSI on udp://{self.host}:{self.port}")

    def stop(self) -> None:
        if self.transport:
            self.transport.close()
            self.transport = None

    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        for line in data.decode("utf-8", errors="replace").splitlines():
            if line.strip():
                self.feed_line(line)

    def feed_line(self, line: str) -> None:
        """Parse a row and append it to the window if it came from the configured transmitter."""
        self.stats["received"] += 1
        try:
            frame = parse_esp32_csi_line(line)
        except ESP32CSIParseError as e:
            self.stats["parse_errors"] += 1
            logger.debug(f"Skipping ESP32 row: {e}")
            return

        if self.transmitter_mac and frame.mac != self.transmitter_mac:
            self.stats["other_transmitters"] += 1
            return

        self.frames.append(frame)
        self.stats["accepted"] += 1
        self.last_frame_time = time.monotonic()

    def get_window(self) -> Optional[np.ndarray]:
        """Return buffered CSI as complex array (antennas=1, subcarriers, samples), or None if empty."""
        if not self.frames:
            return None
        return np.stack([f.csi for f in self.frames], axis=-1)[np.newaxis, ...]

    def seconds_since_last_frame(self) -> Optional[float]:
        if self.last_frame_time is None:
            return None
        return time.monotonic() - self.last_frame_time
