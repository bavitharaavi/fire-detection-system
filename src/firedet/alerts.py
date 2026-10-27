"""Alert dispatch with cooldown so one fire doesn't send hundreds of messages."""
from __future__ import annotations

import logging
import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional, Protocol

import cv2
import numpy as np

from .fusion import Assessment, Level

log = logging.getLogger("firedet.alerts")


class Notifier(Protocol):
    def send(self, subject: str, body: str, image_path: Optional[str]) -> None: ...


class ConsoleNotifier:
    def send(self, subject, body, image_path=None):
        print(f"\n🔥 {subject}\n{body}\n" + (f"snapshot: {image_path}\n" if image_path else ""))


class EmailNotifier:
    """SMTP e-mail. Credentials come from env vars, never from code/config files."""

    def __init__(self, to: List[str], host: str = "smtp.gmail.com", port: int = 587):
        self.to, self.host, self.port = to, host, port
        self.user = os.environ["FIREDET_SMTP_USER"]
        self.password = os.environ["FIREDET_SMTP_PASSWORD"]  # use an app password

    def send(self, subject, body, image_path=None):
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, self.user, ", ".join(self.to)
        msg.set_content(body)
        if image_path:
            msg.add_attachment(Path(image_path).read_bytes(), maintype="image", subtype="jpeg",
                               filename=Path(image_path).name)
        with smtplib.SMTP(self.host, self.port, timeout=10) as s:
            s.starttls()
            s.login(self.user, self.password)
            s.send_message(msg)


class TelegramNotifier:
    """Telegram bot message + photo. Set FIREDET_TG_TOKEN and FIREDET_TG_CHAT_ID."""

    def __init__(self):
        self.token = os.environ["FIREDET_TG_TOKEN"]
        self.chat_id = os.environ["FIREDET_TG_CHAT_ID"]

    def send(self, subject, body, image_path=None):
        import requests

        base = f"https://api.telegram.org/bot{self.token}"
        text = f"*{subject}*\n{body}"
        if image_path:
            with open(image_path, "rb") as f:
                requests.post(f"{base}/sendPhoto", data={"chat_id": self.chat_id, "caption": text,
                                                         "parse_mode": "Markdown"},
                              files={"photo": f}, timeout=10)
        else:
            requests.post(f"{base}/sendMessage", data={"chat_id": self.chat_id, "text": text,
                                                       "parse_mode": "Markdown"}, timeout=10)


class AlertManager:
    """Sends alerts when the risk level rises, with per-level cooldown.

    - Each level (WARNING, ALARM) has its own cooldown, so escalating to ALARM
      notifies immediately even right after a WARNING.
    - Repeats at the same level are suppressed for `cooldown_s`.
    - The buzzer (via sensor MCU) follows the ALARM state.
    """

    def __init__(self, notifiers: List[Notifier], cooldown_s: float = 60.0,
                 snapshot_dir: str = "alerts", location: str = "Camera 1", sensor=None,
                 clock=time.time):
        self.notifiers = notifiers
        self.cooldown_s = cooldown_s
        self.snapshot_dir = Path(snapshot_dir)
        self.location = location
        self.sensor = sensor
        self.clock = clock
        self._last_level = Level.NORMAL
        self._last_sent = {Level.WARNING: -1e18, Level.ALARM: -1e18}
        self.sent_count = 0

    def _snapshot(self, frame: Optional[np.ndarray]) -> Optional[str]:
        if frame is None:
            return None
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / time.strftime("fire_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(str(path), frame)
        return str(path)

    def handle(self, a: Assessment, frame: Optional[np.ndarray] = None) -> bool:
        """Process one assessment. Returns True if a notification was sent."""
        now = self.clock()
        if self.sensor is not None and (a.level == Level.ALARM) != (self._last_level == Level.ALARM):
            self.sensor.command("BUZZER_ON" if a.level == Level.ALARM else "BUZZER_OFF")
        self._last_level = a.level

        if a.level == Level.NORMAL:
            return False
        # Cooldown is tracked per level, so WARNING -> ALARM still notifies at once,
        # but a score hovering around a threshold can't spam the same level.
        if now - self._last_sent[a.level] < self.cooldown_s:
            return False

        self._last_sent[a.level] = now
        image = self._snapshot(frame)
        subject = f"{a.level.name}: possible fire at {self.location}"
        body = (f"Risk score {a.score:.2f} (vision {a.vision_score:.2f}, sensors {a.sensor_score:.2f})\n"
                f"Evidence: {', '.join(a.reasons) or 'n/a'}\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        for n in self.notifiers:
            try:
                n.send(subject, body, image)
            except Exception as exc:  # noqa: BLE001 - one failing channel must not block others
                log.error("notifier %s failed: %s", type(n).__name__, exc)
        self.sent_count += 1
        return True
