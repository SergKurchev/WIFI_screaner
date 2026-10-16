#!/usr/bin/env python3
"""Forward ESP32-CSI-Tool serial output to the WiFi-DensePose API over UDP.

Runs on the host (Docker Desktop on Windows cannot pass USB serial into the container).
Optionally sends small UDP packets to the ESP32 so the router keeps transmitting frames
to it: every such frame yields one CSI measurement, so --traffic-rate sets the CSI rate.
Without it the ESP32 only sees beacons (~10 per second).

    pip install pyserial
    python scripts/esp32_serial_bridge.py --port COM5
    python scripts/esp32_serial_bridge.py --port COM5 --traffic-to 192.168.8.150 --traffic-rate 100
"""

import argparse
import socket
import sys
import threading
import time


def generate_traffic(target: str, rate: float, stop: threading.Event) -> None:
    """Send tiny UDP packets to the ESP32; the router relays them over the air."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / rate
    next_send = time.perf_counter()
    while not stop.is_set():
        sock.sendto(b"x", (target, 9))  # discard port; the ESP32 just drops it
        next_send += interval
        time.sleep(max(0.0, next_send - time.perf_counter()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="ESP32 serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600, help="baud rate set in ESP32-CSI-Tool menuconfig")
    parser.add_argument("--udp-host", default="127.0.0.1", help="where the API listens for CSI rows")
    parser.add_argument("--udp-port", type=int, default=5500)
    parser.add_argument("--traffic-to", help="ESP32 IP address to send traffic to (raises CSI rate)")
    parser.add_argument("--traffic-rate", type=float, default=100.0, help="packets per second")
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print("pyserial is required: pip install pyserial", file=sys.stderr)
        return 1

    stop = threading.Event()
    if args.traffic_to:
        threading.Thread(target=generate_traffic, args=(args.traffic_to, args.traffic_rate, stop),
                         daemon=True).start()

    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    forwarded = 0
    last_report = time.time()
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        print(f"Forwarding {args.port} -> udp://{args.udp_host}:{args.udp_port} (Ctrl+C to stop)")
        try:
            while True:
                line = ser.readline()
                if line.startswith(b"CSI_DATA"):
                    out.sendto(line, (args.udp_host, args.udp_port))
                    forwarded += 1
                if time.time() - last_report >= 5:
                    print(f"{forwarded / (time.time() - last_report):.1f} CSI rows/s")
                    forwarded, last_report = 0, time.time()
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
