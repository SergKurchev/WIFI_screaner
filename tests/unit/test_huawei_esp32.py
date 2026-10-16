"""Tests for the Huawei HiLink router + ESP32 CSI receiver adaptation."""

import asyncio
import socket

import httpx
import numpy as np
import pytest

from src.hardware.esp32_csi import (
    NUM_SUBCARRIERS,
    ESP32CSIParseError,
    ESP32UDPCSISource,
    parse_esp32_csi_line,
)
from src.hardware.huawei_hilink import HiLinkError, HuaweiHiLinkClient

ROUTER_MAC = "14:3c:c3:9d:11:fd"


def make_row(mac=ROUTER_MAC, secondary_channel=0, n_values=128, value_fn=None):
    """Build an ESP32-CSI-Tool row; subcarrier k (buffer position) gets imag=k, real=-k by default."""
    value_fn = value_fn or (lambda pos: (pos % 100, -(pos % 100)))
    buf = []
    for pos in range(n_values // 2):
        imag, real = value_fn(pos)
        buf += [imag, real]
    header = ["CSI_DATA", "STA", mac, "-42", "11", "0", "0", "0", "1", "1", "0", "0", "0", "0",
              "-95", "0", "1", str(secondary_channel), "123456", "0", "52", "0", "0", "0", str(n_values)]
    return ",".join(header) + ",[" + " ".join(map(str, buf)) + " ]"


class TestParseESP32Line:
    def test_parses_metadata(self):
        frame = parse_esp32_csi_line(make_row())
        assert frame.mac == ROUTER_MAC
        assert frame.rssi == -42
        assert frame.channel == 1
        assert frame.local_timestamp_us == 123456

    def test_keeps_52_useful_subcarriers_in_frequency_order(self):
        frame = parse_esp32_csi_line(make_row())
        assert frame.csi.shape == (NUM_SUBCARRIERS,)
        # buffer positions 38..63 hold subcarriers -26..-1, positions 1..26 hold 1..26
        expected_pos = list(range(38, 64)) + list(range(1, 27))
        np.testing.assert_array_equal(frame.csi.imag, expected_pos)
        np.testing.assert_array_equal(frame.csi.real, [-p for p in expected_pos])

    def test_imaginary_comes_first_in_buffer(self):
        frame = parse_esp32_csi_line(make_row(value_fn=lambda pos: (3, 4)))
        assert frame.csi[0] == 4 + 3j

    def test_accepts_esp_csi_quoted_comma_format(self):
        row = make_row().replace("[", "\"[").replace(" ]", "]\"")
        row = row[:row.index("[")] + row[row.index("["):].replace(" ", ",")
        assert parse_esp32_csi_line(row).csi.shape == (NUM_SUBCARRIERS,)

    def test_accepts_longer_buffer_with_ht_ltf(self):
        assert parse_esp32_csi_line(make_row(n_values=384)).csi.shape == (NUM_SUBCARRIERS,)

    @pytest.mark.parametrize("row", [
        "hello",
        "CSI_DATA,STA,14:3c:c3:9d:11:fd,-42",
        make_row(mac="not-a-mac"),
        make_row(secondary_channel=1),
        make_row(n_values=64),
        make_row().replace("[0 0", "[x 0"),
    ])
    def test_rejects_bad_rows(self, row):
        with pytest.raises(ESP32CSIParseError):
            parse_esp32_csi_line(row)


class TestESP32UDPCSISource:
    def test_filters_by_transmitter_mac_and_builds_window(self):
        source = ESP32UDPCSISource("127.0.0.1", 0, ROUTER_MAC.upper(), window_size=3)
        assert source.get_window() is None

        for _ in range(5):
            source.feed_line(make_row())
        source.feed_line(make_row(mac="aa:bb:cc:dd:ee:ff"))
        source.feed_line("garbage")

        window = source.get_window()
        assert window.shape == (1, NUM_SUBCARRIERS, 3)
        assert np.iscomplexobj(window)
        assert source.stats == {"received": 7, "accepted": 5, "parse_errors": 1, "other_transmitters": 1}
        assert source.seconds_since_last_frame() >= 0

    def test_receives_rows_over_udp(self):
        async def scenario():
            source = ESP32UDPCSISource("127.0.0.1", 0, ROUTER_MAC)
            await source.start()
            port = source.transport.get_extra_info("sockname")[1]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(make_row().encode(), ("127.0.0.1", port))
            for _ in range(50):
                if source.frames:
                    break
                await asyncio.sleep(0.01)
            source.stop()
            return source

        source = asyncio.run(scenario())
        assert len(source.frames) == 1


def hilink_transport(responses):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/webserver/SesTokInfo":
            return httpx.Response(200, text="<response><SesInfo>SessionID=abc</SesInfo>"
                                            "<TokInfo>tok</TokInfo></response>")
        assert request.headers["__RequestVerificationToken"] == "tok"
        assert request.headers["Cookie"] == "SessionID=abc"
        return httpx.Response(200, text=responses[request.url.path])
    return httpx.MockTransport(handler)


class TestHuaweiHiLinkClient:
    def test_reads_basic_information(self):
        client = HuaweiHiLinkClient("192.168.8.1", transport=hilink_transport({
            "/api/device/basic_information":
                "<?xml version=\"1.0\"?><response><devicename>B315s-22</devicename>"
                "<SoftwareVersion>21.333.01.00.00</SoftwareVersion></response>",
        }))
        info = asyncio.run(client.get_basic_information())
        assert info["devicename"] == "B315s-22"

    def test_error_response_raises_with_code(self):
        client = HuaweiHiLinkClient("192.168.8.1", transport=hilink_transport({
            "/api/wlan/basic-settings": "<error><code>100003</code><message></message></error>",
        }))
        with pytest.raises(HiLinkError) as exc:
            asyncio.run(client.get_wlan_settings())
        assert exc.value.code == "100003"

    def test_is_alive_when_login_required(self):
        client = HuaweiHiLinkClient("192.168.8.1", transport=hilink_transport({
            "/api/monitoring/status": "<error><code>125002</code></error>",
        }))
        assert asyncio.run(client.is_alive()) is True

    def test_not_alive_when_unreachable(self):
        def handler(request):
            raise httpx.ConnectError("no route")
        client = HuaweiHiLinkClient("192.168.8.1", transport=httpx.MockTransport(handler))
        assert asyncio.run(client.is_alive()) is False

    def test_invalid_xml_raises(self):
        client = HuaweiHiLinkClient("192.168.8.1", transport=hilink_transport({
            "/api/monitoring/status": "<html>",
        }))
        with pytest.raises(HiLinkError):
            asyncio.run(client.get_status())
