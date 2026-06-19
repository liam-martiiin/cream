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

from pathlib import Path

import pytest

from lerobot.scripts import lerobot_register_camera as cmd
from lerobot.utils.camera_registry import CameraRegistry, DiscoveredCamera


def _cam(usb_path: str, capture_node: str | None = "/dev/video0") -> DiscoveredCamera:
    return DiscoveredCamera(
        usb_path=usb_path,
        vid="0c45",
        pid="6366",
        serial="SN0001",
        model="Innomaker-U20CAM-1080p-S1",
        all_nodes=[capture_node] if capture_node else [],
        capture_node=capture_node,
    )


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path: Path):
    """Point HF_LEROBOT_CALIBRATION at a temp dir so save() doesn't touch the real cache."""
    monkeypatch.setattr("lerobot.utils.camera_registry.HF_LEROBOT_CALIBRATION", tmp_path)
    return tmp_path / "cameras.json"


@pytest.fixture
def two_cameras(monkeypatch):
    """Patch discovery to return two unregistered U20CAMs."""
    cams = [
        _cam("usb-0:3:1.0", "/dev/video2"),
        _cam("usb-0:4:1.0", "/dev/video0"),
    ]
    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.discover_cameras", lambda: cams)
    return cams


@pytest.fixture
def fake_picker(monkeypatch):
    """Install a fake picker that returns whichever camera is queued for it."""

    queue = []

    def _set(*choices):
        queue.clear()
        queue.extend(choices)

    def _picker(name, candidates):
        if not queue:
            return None
        return queue.pop(0)

    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.pick_camera", _picker)
    return _set


@pytest.fixture
def fake_input(monkeypatch):
    """Queue successive input() responses for the re-pair confirmation prompt."""

    def _set(responses):
        it = iter(responses)
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(it))

    return _set


def test_no_cameras_returns_failure(monkeypatch, isolated_registry, capsys):
    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.discover_cameras", lambda: [])
    rc = cmd.register_camera("right_overhead")
    assert rc == 1
    assert "No cameras detected" in capsys.readouterr().err


def test_successful_registration(isolated_registry, two_cameras, fake_picker, capsys):
    fake_picker(two_cameras[0])
    rc = cmd.register_camera("right_overhead")
    assert rc == 0
    out = capsys.readouterr().out
    assert "Registered" in out
    reloaded = CameraRegistry.load()
    assert reloaded.find_by_name("right_overhead").usb_path == "usb-0:3:1.0"


def test_picker_cancelled_returns_failure(isolated_registry, two_cameras, fake_picker, capsys):
    fake_picker()  # empty queue → picker returns None
    rc = cmd.register_camera("right_overhead")
    assert rc == 1
    assert "Cancelled" in capsys.readouterr().err
    assert CameraRegistry.load().find_by_name("right_overhead") is None


def test_camera_with_no_capture_node_rejected(isolated_registry, monkeypatch, fake_picker, capsys):
    broken = _cam("usb-0:5:1.0", capture_node=None)
    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.discover_cameras", lambda: [broken])
    fake_picker(broken)
    rc = cmd.register_camera("right_overhead")
    assert rc == 1
    err = capsys.readouterr().err
    assert "no V4L2 capture-capable node" in err


def test_repair_prompt_confirmed(isolated_registry, two_cameras, fake_picker, fake_input):
    # Pre-register the first camera as "old_name".
    reg = CameraRegistry.load()
    reg.register(two_cameras[0], "old_name")
    reg.save()
    # Now pick the same physical camera but bind it to a new name.
    fake_picker(two_cameras[0])
    fake_input(["y"])  # confirm re-pair
    rc = cmd.register_camera("right_overhead")
    assert rc == 0
    reloaded = CameraRegistry.load()
    assert reloaded.find_by_name("old_name") is None
    assert reloaded.find_by_name("right_overhead").usb_path == "usb-0:3:1.0"


def test_repair_prompt_declined_aborts(isolated_registry, two_cameras, fake_picker, fake_input, capsys):
    reg = CameraRegistry.load()
    reg.register(two_cameras[0], "old_name")
    reg.save()
    fake_picker(two_cameras[0])
    fake_input(["n"])  # decline re-pair
    rc = cmd.register_camera("right_overhead")
    assert rc == 1
    reloaded = CameraRegistry.load()
    assert reloaded.find_by_name("old_name") is not None  # old entry preserved
    assert reloaded.find_by_name("right_overhead") is None


def test_force_skips_repair_prompt(isolated_registry, two_cameras, fake_picker, monkeypatch):
    reg = CameraRegistry.load()
    reg.register(two_cameras[0], "old_name")
    reg.save()
    fake_picker(two_cameras[0])
    # input() must NOT be called when --force is set; if it is, the test fails.
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("input() should not be called"))
    rc = cmd.register_camera("right_overhead", force=True)
    assert rc == 0
    assert CameraRegistry.load().find_by_name("right_overhead").usb_path == "usb-0:3:1.0"


def test_headless_flag_sets_env_var(isolated_registry, two_cameras, fake_picker, monkeypatch):
    monkeypatch.delenv("LEROBOT_HEADLESS_PICKER", raising=False)
    fake_picker(two_cameras[0])
    cmd.register_camera("right_overhead", headless=True)
    import os

    assert os.environ.get("LEROBOT_HEADLESS_PICKER") == "1"


def test_registering_same_camera_with_same_name_is_idempotent(isolated_registry, two_cameras, fake_picker):
    fake_picker(two_cameras[0])
    cmd.register_camera("right_overhead")
    fake_picker(two_cameras[0])
    # Same physical camera, same name → no confirmation prompt, no error.
    rc = cmd.register_camera("right_overhead")
    assert rc == 0
    reg = CameraRegistry.load()
    assert len([c for c in reg if c.name == "right_overhead"]) == 1


def _rs_cam(serial: str = "RS999") -> DiscoveredCamera:
    """A discovered RealSense (no capture node; unique serial)."""
    return DiscoveredCamera(
        usb_path="2-1",
        vid="8086",
        pid="0b07",
        serial=serial,
        model="Intel RealSense D435",
        all_nodes=[],
        capture_node=None,
        kind="intelrealsense",
    )


def test_register_realsense_by_serial(isolated_registry, monkeypatch, fake_picker, capsys):
    """A RealSense (no capture node) registers fine and is keyed by serial, not port."""
    rs = _rs_cam(serial="RS999")
    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.discover_cameras", lambda: [rs])
    fake_picker(rs)

    rc = cmd.register_camera("top_depth")

    assert rc == 0  # NOT rejected for lacking a V4L2 capture node
    out = capsys.readouterr().out
    assert "RealSense" in out
    entry = CameraRegistry.load().find_by_name("top_depth")
    assert entry.kind == "intelrealsense"
    assert entry.serial == "RS999"


# --- batch --all mode (assign_names mocked to bypass the snapshot/prompt UI) ---


def test_register_all_registers_multiple_including_realsense(isolated_registry, monkeypatch, capsys):
    webcam = _cam("usb-0:3:1.0", "/dev/video0")
    rs = _rs_cam(serial="RS999")
    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.discover_cameras", lambda: [webcam, rs])
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_register_camera.assign_names",
        lambda cams: [(webcam, "left_arm"), (rs, "top_depth")],
    )

    rc = cmd.register_all()

    assert rc == 0
    reg = CameraRegistry.load()
    assert reg.find_by_name("left_arm").kind == "opencv"
    rs_entry = reg.find_by_name("top_depth")
    assert rs_entry.kind == "intelrealsense" and rs_entry.serial == "RS999"


def test_register_all_rejects_duplicate_names(isolated_registry, monkeypatch, capsys):
    c1 = _cam("usb-0:3:1.0", "/dev/video0")
    c2 = _cam("usb-0:4:1.0", "/dev/video2")
    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.discover_cameras", lambda: [c1, c2])
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_register_camera.assign_names",
        lambda cams: [(c1, "dup"), (c2, "dup")],
    )

    rc = cmd.register_all()

    assert rc == 1
    assert "Duplicate name" in capsys.readouterr().err
    assert len(CameraRegistry.load()) == 0  # nothing saved


def test_register_all_no_names_returns_failure(isolated_registry, monkeypatch, capsys):
    c1 = _cam("usb-0:3:1.0", "/dev/video0")
    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.discover_cameras", lambda: [c1])
    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.assign_names", lambda cams: [])

    rc = cmd.register_all()

    assert rc == 1
    assert "No names entered" in capsys.readouterr().err


def test_register_all_repairs_camera_under_new_name(isolated_registry, monkeypatch):
    c1 = _cam("usb-0:3:1.0", "/dev/video0")
    reg = CameraRegistry.load()
    reg.register(c1, "old_name")
    reg.save()
    monkeypatch.setattr("lerobot.scripts.lerobot_register_camera.discover_cameras", lambda: [c1])
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_register_camera.assign_names", lambda cams: [(c1, "new_name")]
    )

    rc = cmd.register_all()

    assert rc == 0
    reg2 = CameraRegistry.load()
    assert reg2.find_by_name("new_name") is not None
    assert reg2.find_by_name("old_name") is None  # stale binding dropped
    assert len(reg2) == 1


def test_main_name_and_all_are_mutually_exclusive(isolated_registry, monkeypatch):
    monkeypatch.setattr("sys.argv", ["lerobot-register-camera", "--name", "x", "--all"])
    with pytest.raises(SystemExit) as exc_info:
        cmd.main()
    assert exc_info.value.code != 0  # argparse parser.error exits non-zero


def test_main_requires_name_or_all(isolated_registry, monkeypatch):
    monkeypatch.setattr("sys.argv", ["lerobot-register-camera"])
    with pytest.raises(SystemExit) as exc_info:
        cmd.main()
    assert exc_info.value.code != 0
