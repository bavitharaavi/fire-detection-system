"""Environmental sensors: real (Arduino over serial) or simulated.

The Arduino sketch in firmware/ prints one JSON object per line, e.g.
    {"temp_c": 27.4, "smoke_ppm": 180, "flame": 0}
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SensorReading:
    temp_c: Optional[float] = None
    smoke_ppm: Optional[float] = None
    flame: bool = False
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def parse(cls, line: str) -> Optional["SensorReading"]:
        """Parse one JSON line from the microcontroller; returns None on bad input."""
        try:
            d = json.loads(line.strip())
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(d, dict):
            return None
        return cls(
            temp_c=float(d["temp_c"]) if d.get("temp_c") is not None else None,
            smoke_ppm=float(d["smoke_ppm"]) if d.get("smoke_ppm") is not None else None,
            flame=bool(int(d.get("flame", 0))),
        )


class SerialSensor:
    """Reads the latest SensorReading from a serial port on a background thread."""

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 9600):
        import serial  # pyserial

        self._ser = serial.Serial(port, baud, timeout=1)
        self._latest: Optional[SensorReading] = None
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            raw = self._ser.readline().decode("utf-8", errors="ignore")
            reading = SensorReading.parse(raw)
            if reading:
                with self._lock:
                    self._latest = reading

    def read(self) -> Optional[SensorReading]:
        with self._lock:
            r = self._latest
        # Ignore stale data (sensor unplugged / MCU hung).
        return r if r and time.time() - r.timestamp < 5 else None

    def command(self, cmd: str) -> None:
        """Send a command to the MCU, e.g. 'BUZZER_ON' / 'BUZZER_OFF'."""
        self._ser.write((cmd + "\n").encode())

    def close(self):
        self._running = False
        self._ser.close()


class SimulatedSensor:
    """Room-temperature readings with optional injected fire event (for demos/tests)."""

    def __init__(self, fire_after_s: Optional[float] = None, seed: int = 0, clock=None):
        """clock: callable returning seconds; defaults to wall time. The app passes the
        video timestamp so a recorded clip and the simulated sensors stay in sync."""
        self._clock = clock or time.time
        self._start = self._clock()
        self._fire_after = fire_after_s
        self._rng = random.Random(seed)

    def read(self) -> SensorReading:
        on_fire = self._fire_after is not None and self._clock() - self._start >= self._fire_after
        if on_fire:
            return SensorReading(temp_c=55 + self._rng.uniform(0, 10),
                                 smoke_ppm=700 + self._rng.uniform(0, 300), flame=True)
        return SensorReading(temp_c=24 + self._rng.uniform(-1, 1),
                             smoke_ppm=120 + self._rng.uniform(-20, 20), flame=False)

    def command(self, cmd: str) -> None:
        print(f"[sim-sensor] {cmd}")

    def close(self):
        pass
