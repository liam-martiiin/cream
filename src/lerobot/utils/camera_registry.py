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
Camera registry mapping friendly names to physical cameras.

Each entry carries a ``kind``:

- ``"opencv"`` — a UVC/V4L2 webcam, resolved to a ``/dev/video*`` capture node.
- ``"intelrealsense"`` — an Intel RealSense, resolved (via the pyrealsense2 SDK)
  to its globally-unique device **serial number**.

**Why two strategies.** RealSense (and any well-behaved camera) has a unique
serial, so it is identified by serial alone — fully **port-independent**, no
re-pairing when you replug it. But cheap UVC webcams (e.g. the Innomaker U20CAM
line) ship every unit with the same hardcoded ``iSerial = SN0001``, so
``(vid, pid, serial)`` can't tell two of them apart. For those we fall back to a
**compound key** ``(vid, pid, serial, usb_path)`` — the ``usb_path`` is the
physical USB topology (which hub/port), which survives reboots but breaks when a
cable moves; we recover with a user-supplied picker when it goes stale.

UVC resolve logic (``kind == "opencv"``):

  1. Exact ``(vid, pid, serial, usb_path)`` match → return capture node.
  2. ``(vid, pid, serial)`` matches exactly one camera at a different usb_path
     → silently update the registry, return capture node.
  3. ``(vid, pid, serial)`` matches multiple cameras and the registered
     usb_path is gone → call the user-provided picker callback, persist its
     choice as the new usb_path.
  4. No match → raise ``CameraNotConnectedError``.

RealSense resolve logic (``kind == "intelrealsense"``): a connected RealSense
with the registered serial → return that serial (ignores which port it's in);
otherwise raise ``CameraNotConnectedError``.

UVC cameras commonly create multiple ``/dev/video*`` nodes (capture, metadata,
maybe more). Discovery groups them by their shared USB topology path and
detects the capture-capable node via the V4L2 ``VIDIOC_QUERYCAP`` ioctl.
RealSense is discovered separately through the pyrealsense2 SDK (optional
dependency; absence degrades gracefully to "no RealSense found").
"""

from __future__ import annotations

import fcntl
import json
import struct
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

CAMERAS_REGISTRY_FILENAME = "cameras.json"

# Camera backends a registry entry can target.
KIND_OPENCV = "opencv"
KIND_REALSENSE = "intelrealsense"

# Intel's USB vendor id. RealSense devices also enumerate as UVC nodes under this
# vid (with an empty serial), so we drop those from UVC discovery when the SDK has
# already surfaced the RealSense as its own (serial-bearing) entry.
INTEL_VID = "8086"

# V4L2 ioctl: _IOR('V', 0, sizeof(struct v4l2_capability)).
# sizeof = 16(driver) + 32(card) + 32(bus_info) + 4(version)
#        + 4(capabilities) + 4(device_caps) + 12(reserved) = 104 bytes.
_VIDIOC_QUERYCAP = (2 << 30) | (104 << 16) | (ord("V") << 8) | 0
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_DEVICE_CAPS = 0x80000000


class CameraRegistryError(Exception):
    """Base class for camera registry errors."""


class CameraNotConnectedError(CameraRegistryError):
    """Raised when a registered camera can't be found among connected devices."""


class CameraNameConflictError(CameraRegistryError):
    """Raised when a name is already mapped to a different physical camera."""


class NoCaptureNodeError(CameraRegistryError):
    """Raised when a matched camera has no V4L2 capture-capable node."""


@dataclass
class RegisteredCamera:
    """A single entry in the camera registry."""

    name: str
    vid: str
    pid: str
    serial: str
    usb_path: str
    model: str
    registered_at: str  # ISO-8601 UTC
    # Backend this entry targets ("opencv" | "intelrealsense"). Defaults to
    # "opencv" so pre-``kind`` registry files keep loading unchanged.
    kind: str = KIND_OPENCV


@dataclass
class DiscoveredCamera:
    """A UVC camera currently plugged into the host.

    A single physical camera typically owns multiple ``/dev/video*`` nodes
    (usually one capture + one or more metadata streams). ``all_nodes`` lists
    all of them; ``capture_node`` is the one with ``V4L2_CAP_VIDEO_CAPTURE``,
    or ``None`` if the device exposes only metadata.
    """

    usb_path: str
    vid: str
    pid: str
    serial: str
    model: str
    all_nodes: list[str] = field(default_factory=list)
    capture_node: str | None = None
    # "opencv" for UVC/V4L2 cameras, "intelrealsense" for RealSense (no V4L2
    # capture node; identified by its unique serial).
    kind: str = KIND_OPENCV


def _list_video_nodes() -> list[str]:
    """Return ``/dev/video*`` paths in sorted order. Monkeypatched by tests."""
    return sorted(str(p) for p in Path("/dev").glob("video*"))


def _udev_properties(path: str) -> dict[str, str]:
    """Return udev properties for a single video node.

    Uses ``udevadm info --query=property`` so no extra dependency is required.
    Linux-only; macOS users should currently use the ``index_or_path`` config
    field instead of the registry.
    """
    result = subprocess.run(
        ["udevadm", "info", "--query=property", "--name=" + path],
        capture_output=True,
        text=True,
        check=False,
    )
    props: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        props[key] = value
    return props


def _is_capture_node(path: str) -> bool:
    """Open ``path`` and ask V4L2 whether it exposes a capture stream."""
    try:
        with open(path, "rb") as f:
            buf = bytearray(104)
            fcntl.ioctl(f.fileno(), _VIDIOC_QUERYCAP, buf, True)
    except OSError:
        return False
    capabilities = struct.unpack_from("<I", buf, 84)[0]
    device_caps = struct.unpack_from("<I", buf, 88)[0]
    effective = device_caps if (capabilities & _V4L2_CAP_DEVICE_CAPS) else capabilities
    return bool(effective & _V4L2_CAP_VIDEO_CAPTURE)


def discover_uvc_cameras() -> list[DiscoveredCamera]:
    """Find every UVC camera currently plugged in.

    Groups ``/dev/video*`` nodes that share an ``ID_PATH`` into a single
    ``DiscoveredCamera``. Non-USB video devices (e.g. virtual loopback) and
    devices without a vendor ID are excluded.
    """
    nodes = _list_video_nodes()
    per_node_props: dict[str, dict[str, str]] = {}
    for node in nodes:
        props = _udev_properties(node)
        if props.get("ID_VENDOR_ID") and props.get("ID_PATH"):
            per_node_props[node] = props

    groups: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for node, props in per_node_props.items():
        groups[props["ID_PATH"]].append((node, props))

    cameras: list[DiscoveredCamera] = []
    for usb_path, members in groups.items():
        # Take identifying fields from the first node; they're identical across
        # nodes that share a usb path.
        sample = members[0][1]
        all_nodes = sorted(node for node, _ in members)
        capture_node = next((n for n in all_nodes if _is_capture_node(n)), None)
        cameras.append(
            DiscoveredCamera(
                usb_path=usb_path,
                vid=sample.get("ID_VENDOR_ID", ""),
                pid=sample.get("ID_MODEL_ID", ""),
                serial=sample.get("ID_SERIAL_SHORT", ""),
                model=sample.get("ID_MODEL") or sample.get("ID_V4L_PRODUCT") or "Unknown",
                all_nodes=all_nodes,
                capture_node=capture_node,
            )
        )
    cameras.sort(key=lambda c: c.usb_path)
    return cameras


def discover_realsense_cameras() -> list[DiscoveredCamera]:
    """Find every Intel RealSense currently plugged in, via the pyrealsense2 SDK.

    Each RealSense reports a globally-unique serial number, so it's identified by
    serial alone (port-independent; ``capture_node`` is ``None`` — RealSense is
    driven through the SDK, not V4L2). Returns ``[]`` if pyrealsense2 isn't
    installed or no device is present — never raises.
    """
    from lerobot.utils.import_utils import _pyrealsense2_available

    if not _pyrealsense2_available:
        return []
    try:
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

        infos = RealSenseCamera.find_cameras()
    except Exception:
        # No device, SDK/driver hiccup, etc. — treat as "no RealSense found".
        return []

    cameras: list[DiscoveredCamera] = []
    for info in infos:
        cameras.append(
            DiscoveredCamera(
                usb_path=str(info.get("physical_port", "")),
                vid=INTEL_VID,
                pid=str(info.get("product_id", "")),
                serial=str(info.get("id", "")),
                model=str(info.get("name", "Intel RealSense")),
                all_nodes=[],
                capture_node=None,
                kind=KIND_REALSENSE,
            )
        )
    cameras.sort(key=lambda c: c.serial)
    return cameras


def discover_cameras() -> list[DiscoveredCamera]:
    """Discover all supported cameras: UVC/V4L2 webcams plus Intel RealSense.

    When RealSense devices are present, their stray Intel-vid UVC nodes (which
    carry no usable serial) are dropped so each RealSense appears once — as its
    serial-bearing ``intelrealsense`` entry.
    """
    uvc = discover_uvc_cameras()
    realsense = discover_realsense_cameras()
    if realsense:
        uvc = [c for c in uvc if c.vid != INTEL_VID]
    return uvc + realsense


# A picker is a callable invoked when ``resolve()`` can't disambiguate from
# identifiers alone. It receives the friendly name being resolved and the
# list of candidate cameras, and must return the chosen one (or ``None`` if
# the user cancelled).
PickerFn = Callable[[str, list[DiscoveredCamera]], "DiscoveredCamera | None"]


class CameraRegistry:
    """JSON-backed registry mapping friendly names to physical cameras."""

    def __init__(
        self,
        cameras: list[RegisteredCamera] | None = None,
        path: Path | None = None,
    ):
        self._cameras: list[RegisteredCamera] = list(cameras) if cameras else []
        self._path: Path = path if path is not None else (HF_LEROBOT_CALIBRATION / CAMERAS_REGISTRY_FILENAME)

    @classmethod
    def load(cls, path: Path | None = None) -> CameraRegistry:
        """Load the registry from disk. Returns an empty registry if no file exists yet."""
        path = path if path is not None else (HF_LEROBOT_CALIBRATION / CAMERAS_REGISTRY_FILENAME)
        if not path.is_file():
            return cls(cameras=[], path=path)
        with open(path) as f:
            raw = json.load(f)
        cameras = [RegisteredCamera(**entry) for entry in raw.get("cameras", [])]
        return cls(cameras=cameras, path=path)

    def save(self) -> None:
        """Write the registry to disk, creating parent directories if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cameras": [asdict(c) for c in self._cameras]}
        with open(self._path, "w") as f:
            json.dump(payload, f, indent=4)
            f.write("\n")

    @property
    def path(self) -> Path:
        return self._path

    def __iter__(self) -> Iterator[RegisteredCamera]:
        return iter(self._cameras)

    def __len__(self) -> int:
        return len(self._cameras)

    def find_by_name(self, name: str) -> RegisteredCamera | None:
        return next((c for c in self._cameras if c.name == name), None)

    def register(self, camera: DiscoveredCamera, name: str, *, replace: bool = False) -> RegisteredCamera:
        """Bind a friendly name to a discovered camera.

        - If ``name`` is already mapped to a different ``(vid, pid, serial,
          usb_path)`` quadruple:
          - ``replace=False`` (default): raise ``CameraNameConflictError`` so the
            caller can prompt to re-pair.
          - ``replace=True``: re-pair — drop the stale ``name`` → old-camera binding
            and bind ``name`` to this camera instead. Use only once the user has
            confirmed the re-pair.
        - Otherwise add or update the entry.
        """
        existing = self.find_by_name(name)
        if existing is not None and not _matches(existing, camera):
            if not replace:
                raise CameraNameConflictError(
                    f"Name {name!r} is already mapped to a different camera "
                    f"({existing.model} at {existing.usb_path}). "
                    "Unregister that entry first, or choose a different name."
                )
            # Re-pair: the name is moving to a new physical camera. Drop the stale
            # binding so we don't end up with two entries sharing the same name.
            self._cameras.remove(existing)
            existing = None
        entry = RegisteredCamera(
            name=name,
            vid=camera.vid,
            pid=camera.pid,
            serial=camera.serial,
            usb_path=camera.usb_path,
            model=camera.model,
            registered_at=_now_iso(),
            kind=camera.kind,
        )
        if existing is not None:
            existing.usb_path = entry.usb_path
            existing.registered_at = entry.registered_at
            existing.model = entry.model
            existing.kind = entry.kind
            return existing
        self._cameras.append(entry)
        return entry

    def find_by_camera(self, camera: DiscoveredCamera) -> RegisteredCamera | None:
        """Return the entry already bound to this physical camera, if any."""
        return next((c for c in self._cameras if _matches(c, camera)), None)

    def unregister(self, name: str) -> bool:
        """Remove a camera by name. Returns True if removed, False if not found."""
        before = len(self._cameras)
        self._cameras = [c for c in self._cameras if c.name != name]
        return len(self._cameras) < before

    def resolve(self, name: str, picker: PickerFn | None = None) -> str:
        """Resolve a friendly name to the identifier its backend needs.

        - ``kind == "opencv"``: returns a ``/dev/video*`` capture node (and may
          self-heal/persist a moved ``usb_path``); see module docstring.
        - ``kind == "intelrealsense"``: returns the device **serial number**
          (port-independent), to be passed as ``serial_number_or_name``.

        Raises ``CameraNotConnectedError`` if the camera isn't currently present.
        """
        entry = self.find_by_name(name)
        if entry is None:
            raise CameraNotConnectedError(
                f"No camera named {name!r} is registered. "
                "Run `lerobot-register-camera` first, or set --robot.cameras=... index_or_path manually."
            )

        if entry.kind == KIND_REALSENSE:
            return self._resolve_realsense(entry)

        connected = discover_uvc_cameras()

        # Case 1: exact compound-key match.
        for cam in connected:
            if _matches(entry, cam):
                return _require_capture_node(name, cam)

        # Case 2 & 3: same physical model+serial, but usb_path moved.
        candidates = [
            cam
            for cam in connected
            if cam.vid == entry.vid and cam.pid == entry.pid and cam.serial == entry.serial
        ]
        if len(candidates) == 1:
            # Single ambiguity-free candidate at a new port → silent self-heal.
            chosen = candidates[0]
            entry.usb_path = chosen.usb_path
            entry.registered_at = _now_iso()
            self.save()
            return _require_capture_node(name, chosen)

        if len(candidates) > 1:
            if picker is None:
                ports = ", ".join(sorted(c.usb_path for c in candidates))
                raise CameraNotConnectedError(
                    f"Camera {name!r} (serial {entry.serial!r}) is not at its registered USB "
                    f"port, and {len(candidates)} cameras share that serial, so it can't be "
                    f"resolved unambiguously. Candidate ports: {ports}. "
                    f"Plug {name!r} back into its registered port, or re-pair it with "
                    f"`lerobot-register-camera --name {name}` (which lets you pick it from a preview)."
                )
            chosen = picker(name, candidates)
            if chosen is None:
                raise CameraNotConnectedError(f"User cancelled picker for camera {name!r}.")
            entry.usb_path = chosen.usb_path
            entry.registered_at = _now_iso()
            self.save()
            return _require_capture_node(name, chosen)

        # Case 4: nothing matches at all.
        raise CameraNotConnectedError(
            f"Camera {name!r} (model {entry.model!r}, serial {entry.serial!r}) is "
            f"registered but not currently connected. Plug it in and try again."
        )

    def _resolve_realsense(self, entry: RegisteredCamera) -> str:
        """Resolve a RealSense entry to its serial — independent of which port it's in."""
        for cam in discover_realsense_cameras():
            if cam.serial == entry.serial:
                return entry.serial
        raise CameraNotConnectedError(
            f"RealSense camera {entry.name!r} (serial {entry.serial!r}) is registered but not "
            "currently connected. Plug it into any USB port and try again. "
            "(If pyrealsense2 isn't installed, install `lerobot[intelrealsense]`.)"
        )


def _matches(entry: RegisteredCamera, cam: DiscoveredCamera) -> bool:
    if entry.kind != cam.kind:
        return False
    if entry.kind == KIND_REALSENSE:
        # RealSense serials are globally unique — port-independent identity.
        return cam.serial == entry.serial
    return (
        cam.vid == entry.vid
        and cam.pid == entry.pid
        and cam.serial == entry.serial
        and cam.usb_path == entry.usb_path
    )


def _require_capture_node(name: str, cam: DiscoveredCamera) -> str:
    if cam.capture_node is None:
        raise NoCaptureNodeError(
            f"Camera {name!r} ({cam.model} at {cam.usb_path}) has no V4L2 "
            "capture-capable node. The device may expose only metadata streams."
        )
    return cam.capture_node


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
