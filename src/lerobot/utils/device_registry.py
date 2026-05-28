# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Device registry mapping physical USB serial numbers to friendly robot names.

Each entry binds a USB-to-serial board's factory-burned hardware serial number
(exposed cross-platform by pyserial as ``ListPortInfo.serial_number``) to a
user-chosen name and a robot type. This lets LeRobot:

1. Auto-resolve a port path at connect time, so users don't pass
   ``--robot.port=/dev/ttyACM0`` flags that change on every replug.
2. Detect when a different physical board has been plugged in under the same
   name, preventing silent calibration mismatch.

The registry is stored as JSON at ``$HF_LEROBOT_CALIBRATION/devices.json`` so
it lives alongside the calibration files it relates to.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

# Waveshare bus-servo board for SO-101 uses the WCH CH343 USB-to-serial chip.
SO101_BOARD_VID = 0x1A86
SO101_BOARD_PID = 0x55D3

DEVICES_REGISTRY_FILENAME = "devices.json"


class DeviceRegistryError(Exception):
    """Base class for device registry errors."""


class DeviceNotConnectedError(DeviceRegistryError):
    """Raised when a registered device is not currently plugged in."""


class DeviceNameConflictError(DeviceRegistryError):
    """Raised when a name is already mapped to a different serial."""


class DeviceMismatchError(DeviceRegistryError):
    """Raised when a connected board's serial doesn't match the registry."""

    def __init__(self, name: str, expected_serial: str, found_serial: str):
        self.name = name
        self.expected_serial = expected_serial
        self.found_serial = found_serial
        super().__init__(
            f"Robot {name!r} is registered to board serial {expected_serial!r}, "
            f"but the board connected for it has serial {found_serial!r}. "
            "Did you swap USB cables? Re-pair with `lerobot-register-device`, "
            "or plug in the correct board."
        )


@dataclass
class RegisteredDevice:
    """A single entry in the device registry."""

    serial: str
    name: str
    robot_type: str
    registered_at: str  # ISO-8601 UTC


@dataclass
class ConnectedBoard:
    """A USB serial board currently plugged into the host."""

    port: str
    serial_number: str


def _comports():
    """Return the list of currently connected serial ports (lazy pyserial import)."""
    from lerobot.utils.import_utils import require_package

    require_package("pyserial", extra="hardware", import_name="serial")
    from serial.tools import list_ports

    return list_ports.comports()


def list_connected_boards() -> list[ConnectedBoard]:
    """Scan USB serial devices and return SO-101 boards only.

    Cross-platform: pyserial abstracts ``/dev/ttyACM*`` (Linux) vs
    ``/dev/cu.usbmodem*`` (macOS). Boards without a readable serial number
    are skipped — rare, but possible on counterfeit chips.
    """
    boards: list[ConnectedBoard] = []
    for info in _comports():
        if info.vid != SO101_BOARD_VID or info.pid != SO101_BOARD_PID:
            continue
        if not info.serial_number:
            continue
        boards.append(ConnectedBoard(port=info.device, serial_number=info.serial_number))
    return boards


def serial_for_port(port: str) -> str | None:
    """Return the hardware serial of the board at ``port``, or ``None`` if not found."""
    for info in _comports():
        if info.device == port:
            return info.serial_number
    return None


class DeviceRegistry:
    """JSON-backed registry mapping board serial numbers to friendly robot names."""

    def __init__(
        self,
        devices: list[RegisteredDevice] | None = None,
        path: Path | None = None,
    ):
        self._devices: list[RegisteredDevice] = list(devices) if devices else []
        self._path: Path = path if path is not None else (HF_LEROBOT_CALIBRATION / DEVICES_REGISTRY_FILENAME)

    @classmethod
    def load(cls, path: Path | None = None) -> DeviceRegistry:
        """Load the registry from disk. Returns an empty registry if no file exists yet."""
        path = path if path is not None else (HF_LEROBOT_CALIBRATION / DEVICES_REGISTRY_FILENAME)
        if not path.is_file():
            return cls(devices=[], path=path)
        with open(path) as f:
            raw = json.load(f)
        devices = [RegisteredDevice(**entry) for entry in raw.get("devices", [])]
        return cls(devices=devices, path=path)

    def save(self) -> None:
        """Write the registry to disk, creating parent directories if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"devices": [asdict(d) for d in self._devices]}
        with open(self._path, "w") as f:
            json.dump(payload, f, indent=4)
            f.write("\n")

    @property
    def path(self) -> Path:
        return self._path

    def __iter__(self) -> Iterator[RegisteredDevice]:
        return iter(self._devices)

    def __len__(self) -> int:
        return len(self._devices)

    def find_by_name(self, name: str) -> RegisteredDevice | None:
        return next((d for d in self._devices if d.name == name), None)

    def find_by_serial(self, serial: str) -> RegisteredDevice | None:
        return next((d for d in self._devices if d.serial == serial), None)

    def register(self, serial: str, name: str, robot_type: str) -> RegisteredDevice:
        """Add or update a device entry.

        - If ``serial`` already exists: update name and robot_type in place.
        - If ``name`` already exists for a different serial: raise
          ``DeviceNameConflictError`` so the caller can prompt the user.
        """
        existing_by_name = self.find_by_name(name)
        if existing_by_name is not None and existing_by_name.serial != serial:
            raise DeviceNameConflictError(
                f"Name {name!r} is already mapped to serial {existing_by_name.serial!r}. "
                "Unregister that entry first, or choose a different name."
            )
        existing_by_serial = self.find_by_serial(serial)
        if existing_by_serial is not None:
            existing_by_serial.name = name
            existing_by_serial.robot_type = robot_type
            existing_by_serial.registered_at = _now_iso()
            return existing_by_serial
        device = RegisteredDevice(
            serial=serial,
            name=name,
            robot_type=robot_type,
            registered_at=_now_iso(),
        )
        self._devices.append(device)
        return device

    def unregister(self, name: str) -> bool:
        """Remove a device by name. Returns True if removed, False if not found."""
        before = len(self._devices)
        self._devices = [d for d in self._devices if d.name != name]
        return len(self._devices) < before

    def resolve_port(self, name: str) -> str:
        """Find the device path of the registered board named ``name``.

        Raises ``DeviceNotConnectedError`` if the name is unknown or the
        corresponding board is not currently plugged in.
        """
        device = self.find_by_name(name)
        if device is None:
            raise DeviceNotConnectedError(
                f"No device named {name!r} is registered. "
                "Run `lerobot-register-device` first, or pass --robot.port manually."
            )
        for board in list_connected_boards():
            if board.serial_number == device.serial:
                return board.port
        raise DeviceNotConnectedError(
            f"Device {name!r} (serial {device.serial!r}) is registered but not currently "
            "connected. Plug in the board and try again."
        )


def resolve_or_verify_port(name: str, port: str | None, register_command_hint: str | None = None) -> str:
    """Resolve a robot's port from the registry, or verify a user-supplied one.

    Intended to be called from robot/teleoperator ``__init__`` to abstract over
    the four legitimate states:

    +----------------+----------------------+-----------------------------------+
    | port           | name registered?     | result                            |
    +================+======================+===================================+
    | ``None``       | yes                  | return registry-resolved port     |
    +----------------+----------------------+-----------------------------------+
    | ``None``       | no                   | raise: user must register or pass |
    |                |                      | ``--robot.port``                  |
    +----------------+----------------------+-----------------------------------+
    | given          | yes                  | verify the connected board's      |
    |                |                      | serial matches; raise on mismatch |
    +----------------+----------------------+-----------------------------------+
    | given          | no                   | return port unchanged (legacy)    |
    +----------------+----------------------+-----------------------------------+

    ``register_command_hint`` is woven into error messages so the suggested
    ``lerobot-register-device`` invocation includes the right ``--type``.
    """
    registry = DeviceRegistry.load()
    registered = registry.find_by_name(name)

    if port is None:
        if registered is None:
            hint = f" {register_command_hint}" if register_command_hint else ""
            raise ValueError(
                f"No port specified for id={name!r}, and no device named {name!r} is "
                f"registered. Either pass --robot.port, or run "
                f"`lerobot-register-device{hint} --name {name}`."
            )
        return registry.resolve_port(name)

    if registered is not None:
        connected_serial = serial_for_port(port)
        if connected_serial is not None and connected_serial != registered.serial:
            raise DeviceMismatchError(name, registered.serial, connected_serial)
    return port


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
