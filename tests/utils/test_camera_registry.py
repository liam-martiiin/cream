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

import json
from pathlib import Path

import pytest

from lerobot.utils.camera_registry import (
    CameraNameConflictError,
    CameraNotConnectedError,
    CameraRegistry,
    DiscoveredCamera,
    NoCaptureNodeError,
    discover_uvc_cameras,
)

# --- Fakes for the three monkeypatch seams ----------------------------------


def _props(
    vid: str = "0c45",
    pid: str = "6366",
    serial: str = "SN0001",
    usb_path: str = "pci-0000:80:14.0-usb-0:4:1.0",
    model: str = "Innomaker-U20CAM-1080p-S1",
) -> dict[str, str]:
    return {
        "ID_VENDOR_ID": vid,
        "ID_MODEL_ID": pid,
        "ID_SERIAL_SHORT": serial,
        "ID_PATH": usb_path,
        "ID_MODEL": model,
    }


@pytest.fixture
def fake_discovery(monkeypatch):
    """Inject a fake hardware view: video-node list, udev properties, capture-node check."""

    def _set(*, nodes: list[str], properties: dict[str, dict[str, str]], capture: dict[str, bool]):
        monkeypatch.setattr("lerobot.utils.camera_registry._list_video_nodes", lambda: list(nodes))
        monkeypatch.setattr(
            "lerobot.utils.camera_registry._udev_properties",
            lambda path: properties.get(path, {}),
        )
        monkeypatch.setattr(
            "lerobot.utils.camera_registry._is_capture_node",
            lambda path: capture.get(path, False),
        )

    return _set


@pytest.fixture
def two_u20cams(fake_discovery):
    """Two identical U20CAMs, each with a capture node + a metadata node."""
    fake_discovery(
        nodes=["/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3"],
        properties={
            "/dev/video0": _props(usb_path="usb-0:4:1.0"),
            "/dev/video1": _props(usb_path="usb-0:4:1.0"),  # same camera, metadata
            "/dev/video2": _props(usb_path="usb-0:3:1.0"),
            "/dev/video3": _props(usb_path="usb-0:3:1.0"),  # same camera, metadata
        },
        capture={
            "/dev/video0": True,
            "/dev/video1": False,
            "/dev/video2": True,
            "/dev/video3": False,
        },
    )


@pytest.fixture
def registry(tmp_path: Path) -> CameraRegistry:
    return CameraRegistry(path=tmp_path / "cameras.json")


# --- Discovery --------------------------------------------------------------


def test_discovery_groups_multi_node_cameras(two_u20cams):
    cameras = discover_uvc_cameras()
    assert len(cameras) == 2
    by_path = {c.usb_path: c for c in cameras}
    cam_a = by_path["usb-0:3:1.0"]
    cam_b = by_path["usb-0:4:1.0"]
    assert cam_a.all_nodes == ["/dev/video2", "/dev/video3"]
    assert cam_a.capture_node == "/dev/video2"
    assert cam_b.all_nodes == ["/dev/video0", "/dev/video1"]
    assert cam_b.capture_node == "/dev/video0"


def test_discovery_skips_non_usb_devices(fake_discovery):
    # A loopback / virtual device has no ID_VENDOR_ID, so it should be excluded.
    fake_discovery(
        nodes=["/dev/video0", "/dev/video10"],
        properties={
            "/dev/video0": _props(),
            "/dev/video10": {},  # virtual device, no udev props
        },
        capture={"/dev/video0": True, "/dev/video10": True},
    )
    cameras = discover_uvc_cameras()
    assert len(cameras) == 1
    assert cameras[0].all_nodes == ["/dev/video0"]


def test_discovery_marks_no_capture_node_when_only_metadata(fake_discovery):
    fake_discovery(
        nodes=["/dev/video0"],
        properties={"/dev/video0": _props()},
        capture={"/dev/video0": False},
    )
    cameras = discover_uvc_cameras()
    assert cameras[0].capture_node is None


# --- Registry CRUD ----------------------------------------------------------


def test_load_returns_empty_when_file_missing(tmp_path: Path):
    reg = CameraRegistry.load(path=tmp_path / "nope.json")
    assert len(reg) == 0


def test_register_adds_new_camera(registry: CameraRegistry):
    cam = DiscoveredCamera(
        usb_path="usb-0:4:1.0",
        vid="0c45",
        pid="6366",
        serial="SN0001",
        model="Innomaker-U20CAM-1080p-S1",
        all_nodes=["/dev/video0", "/dev/video1"],
        capture_node="/dev/video0",
    )
    entry = registry.register(cam, "right_overhead")
    assert entry.name == "right_overhead"
    assert entry.usb_path == "usb-0:4:1.0"
    assert entry.registered_at  # timestamp populated
    assert len(registry) == 1


def test_register_raises_on_name_conflict(registry: CameraRegistry):
    cam_a = DiscoveredCamera(
        usb_path="usb-0:4:1.0",
        vid="0c45",
        pid="6366",
        serial="SN0001",
        model="U20CAM",
        capture_node="/dev/video0",
    )
    cam_b = DiscoveredCamera(
        usb_path="usb-0:3:1.0",
        vid="0c45",
        pid="6366",
        serial="SN0001",
        model="U20CAM",
        capture_node="/dev/video2",
    )
    registry.register(cam_a, "right_overhead")
    with pytest.raises(CameraNameConflictError, match="right_overhead"):
        registry.register(cam_b, "right_overhead")


def test_register_same_name_same_camera_updates_in_place(registry: CameraRegistry):
    cam = DiscoveredCamera(
        usb_path="usb-0:4:1.0",
        vid="0c45",
        pid="6366",
        serial="SN0001",
        model="U20CAM",
        capture_node="/dev/video0",
    )
    registry.register(cam, "right_overhead")
    first_ts = registry.find_by_name("right_overhead").registered_at
    # Re-register the same physical camera — should update timestamp, not raise.
    registry.register(cam, "right_overhead")
    assert len(registry) == 1
    assert registry.find_by_name("right_overhead").registered_at >= first_ts


def test_unregister_removes_by_name(registry: CameraRegistry):
    cam = DiscoveredCamera(
        usb_path="usb-0:4:1.0",
        vid="0c45",
        pid="6366",
        serial="SN0001",
        model="U20CAM",
        capture_node="/dev/video0",
    )
    registry.register(cam, "right_overhead")
    assert registry.unregister("right_overhead") is True
    assert len(registry) == 0
    assert registry.unregister("right_overhead") is False  # idempotent


def test_save_and_load_round_trip(tmp_path: Path):
    path = tmp_path / "cameras.json"
    reg = CameraRegistry(path=path)
    reg.register(
        DiscoveredCamera(
            usb_path="usb-0:4:1.0",
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="U20CAM",
            capture_node="/dev/video0",
        ),
        "right_overhead",
    )
    reg.save()

    assert path.is_file()
    raw = json.loads(path.read_text())
    assert raw["cameras"][0]["name"] == "right_overhead"

    reloaded = CameraRegistry.load(path=path)
    assert len(reloaded) == 1
    assert reloaded.find_by_name("right_overhead").usb_path == "usb-0:4:1.0"


def test_save_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "cameras.json"
    reg = CameraRegistry(path=nested)
    reg.register(
        DiscoveredCamera(
            usb_path="usb-0:4:1.0",
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="U20CAM",
            capture_node="/dev/video0",
        ),
        "right_overhead",
    )
    reg.save()
    assert nested.is_file()


# --- resolve() --------------------------------------------------------------


def _register_two_u20cams(registry: CameraRegistry):
    """Pre-populate the registry with two cameras that match the two_u20cams fixture."""
    registry.register(
        DiscoveredCamera(
            usb_path="usb-0:4:1.0",
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="Innomaker-U20CAM-1080p-S1",
            capture_node="/dev/video0",
        ),
        "right_overhead",
    )
    registry.register(
        DiscoveredCamera(
            usb_path="usb-0:3:1.0",
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="Innomaker-U20CAM-1080p-S1",
            capture_node="/dev/video2",
        ),
        "left_overhead",
    )


def test_resolve_happy_path(registry: CameraRegistry, two_u20cams):
    _register_two_u20cams(registry)
    assert registry.resolve("right_overhead") == "/dev/video0"
    assert registry.resolve("left_overhead") == "/dev/video2"


def test_resolve_unknown_name_raises(registry: CameraRegistry, two_u20cams):
    with pytest.raises(CameraNotConnectedError, match="No camera named"):
        registry.resolve("ghost")


def test_resolve_silent_self_heal_when_single_candidate_moved(registry: CameraRegistry, fake_discovery):
    # Register at usb-0:4:1.0.
    registry.register(
        DiscoveredCamera(
            usb_path="usb-0:4:1.0",
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="U20CAM",
            capture_node="/dev/video0",
        ),
        "right_overhead",
    )
    registry.save()
    # Now only one U20CAM is plugged in, at a different port. No ambiguity.
    fake_discovery(
        nodes=["/dev/video5"],
        properties={"/dev/video5": _props(usb_path="usb-0:7:1.0")},
        capture={"/dev/video5": True},
    )
    resolved = registry.resolve("right_overhead")
    assert resolved == "/dev/video5"
    # Registry should have been updated to the new usb_path.
    reloaded = CameraRegistry.load(path=registry.path)
    assert reloaded.find_by_name("right_overhead").usb_path == "usb-0:7:1.0"


def test_resolve_picker_invoked_on_ambiguous_match(registry: CameraRegistry, two_u20cams):
    # Register at a usb_path that's no longer occupied.
    registry.register(
        DiscoveredCamera(
            usb_path="usb-0:9:1.0",  # not in the connected set
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="Innomaker-U20CAM-1080p-S1",
            capture_node="/dev/video0",
        ),
        "right_overhead",
    )
    registry.save()
    picker_calls = []

    def picker(name, candidates):
        picker_calls.append((name, [c.usb_path for c in candidates]))
        # User picks the camera at usb-0:3:1.0.
        return next(c for c in candidates if c.usb_path == "usb-0:3:1.0")

    resolved = registry.resolve("right_overhead", picker=picker)
    assert resolved == "/dev/video2"
    assert picker_calls == [("right_overhead", ["usb-0:3:1.0", "usb-0:4:1.0"])]
    # Registry persisted the new usb_path.
    reloaded = CameraRegistry.load(path=registry.path)
    assert reloaded.find_by_name("right_overhead").usb_path == "usb-0:3:1.0"


def test_resolve_picker_cancelled_raises(registry: CameraRegistry, two_u20cams):
    registry.register(
        DiscoveredCamera(
            usb_path="usb-0:9:1.0",
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="U20CAM",
            capture_node="/dev/video0",
        ),
        "right_overhead",
    )
    with pytest.raises(CameraNotConnectedError, match="cancelled"):
        registry.resolve("right_overhead", picker=lambda name, candidates: None)


def test_resolve_ambiguous_without_picker_raises(registry: CameraRegistry, two_u20cams):
    registry.register(
        DiscoveredCamera(
            usb_path="usb-0:9:1.0",
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="U20CAM",
            capture_node="/dev/video0",
        ),
        "right_overhead",
    )
    with pytest.raises(CameraNotConnectedError, match="picker"):
        registry.resolve("right_overhead", picker=None)


def test_resolve_no_match_at_all_raises(registry: CameraRegistry, fake_discovery):
    registry.register(
        DiscoveredCamera(
            usb_path="usb-0:4:1.0",
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="U20CAM",
            capture_node="/dev/video0",
        ),
        "right_overhead",
    )
    fake_discovery(nodes=[], properties={}, capture={})  # camera unplugged
    with pytest.raises(CameraNotConnectedError, match="not currently connected"):
        registry.resolve("right_overhead")


def test_resolve_raises_when_matched_camera_has_no_capture_node(registry: CameraRegistry, fake_discovery):
    registry.register(
        DiscoveredCamera(
            usb_path="usb-0:4:1.0",
            vid="0c45",
            pid="6366",
            serial="SN0001",
            model="U20CAM",
            capture_node="/dev/video0",
        ),
        "right_overhead",
    )
    # Same camera plugged in, but the kernel only exposed a metadata node.
    fake_discovery(
        nodes=["/dev/video1"],
        properties={"/dev/video1": _props(usb_path="usb-0:4:1.0")},
        capture={"/dev/video1": False},
    )
    with pytest.raises(NoCaptureNodeError):
        registry.resolve("right_overhead")
