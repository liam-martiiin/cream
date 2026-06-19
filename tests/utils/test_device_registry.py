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
from types import SimpleNamespace

import pytest

from lerobot.utils.device_registry import (
    SO101_BOARD_PID,
    SO101_BOARD_VID,
    DeviceMismatchError,
    DeviceNameConflictError,
    DeviceNotConnectedError,
    DeviceRegistry,
    list_connected_boards,
    resolve_or_verify_port,
    serial_for_port,
)


def _fake_port(device: str, vid: int | None, pid: int | None, serial_number: str | None):
    """Build a fake ``ListPortInfo``-like object for mocking ``comports()``."""
    return SimpleNamespace(device=device, vid=vid, pid=pid, serial_number=serial_number)


@pytest.fixture
def fake_comports(monkeypatch):
    """Return a setter that overrides ``serial.tools.list_ports.comports`` with a fixed list."""

    def _set(ports):
        monkeypatch.setattr("lerobot.utils.device_registry._comports", lambda: list(ports))

    return _set


@pytest.fixture
def two_so101_boards(fake_comports):
    """Two SO-101 boards plus some unrelated USB noise."""
    fake_comports(
        [
            _fake_port("/dev/ttyACM0", SO101_BOARD_VID, SO101_BOARD_PID, "AAA111"),
            _fake_port("/dev/ttyACM1", SO101_BOARD_VID, SO101_BOARD_PID, "BBB222"),
            _fake_port("/dev/ttyS0", None, None, None),  # built-in serial, no USB
            _fake_port("/dev/ttyUSB0", 0x1234, 0x5678, "ZZZ999"),  # different chip
        ]
    )


@pytest.fixture
def registry(tmp_path: Path) -> DeviceRegistry:
    return DeviceRegistry(path=tmp_path / "devices.json")


def test_load_returns_empty_when_file_missing(tmp_path: Path):
    reg = DeviceRegistry.load(path=tmp_path / "nope.json")
    assert len(reg) == 0


def test_register_adds_new_device(registry: DeviceRegistry):
    device = registry.register(serial="AAA111", name="leader", robot_type="so101_leader")
    assert device.serial == "AAA111"
    assert device.name == "leader"
    assert device.robot_type == "so101_leader"
    assert device.registered_at  # timestamp populated
    assert len(registry) == 1


def test_register_updates_existing_serial(registry: DeviceRegistry):
    registry.register(serial="AAA111", name="leader", robot_type="so101_leader")
    updated = registry.register(serial="AAA111", name="left_leader", robot_type="so101_leader")
    assert len(registry) == 1
    assert updated.name == "left_leader"
    assert registry.find_by_name("leader") is None
    assert registry.find_by_name("left_leader") is not None


def test_register_raises_on_name_conflict(registry: DeviceRegistry):
    registry.register(serial="AAA111", name="leader", robot_type="so101_leader")
    with pytest.raises(DeviceNameConflictError, match="leader"):
        registry.register(serial="BBB222", name="leader", robot_type="so101_leader")


def test_register_replace_repairs_name_to_new_serial(registry: DeviceRegistry):
    # Re-pair: 'follower' moves from its old board to a freshly plugged one.
    registry.register(serial="OLD111", name="follower", robot_type="so101_follower")
    updated = registry.register(serial="NEW222", name="follower", robot_type="so101_follower", replace=True)
    assert len(registry) == 1  # no duplicate name left behind
    assert updated.serial == "NEW222"
    assert registry.find_by_name("follower").serial == "NEW222"
    assert registry.find_by_serial("OLD111") is None


def test_register_replace_merges_when_new_serial_already_registered(registry: DeviceRegistry):
    # The new board is itself already registered under another name. Re-pairing
    # 'follower' onto it must leave a single, consistent entry.
    registry.register(serial="OLD111", name="follower", robot_type="so101_follower")
    registry.register(serial="NEW222", name="spare", robot_type="so101_follower")
    registry.register(serial="NEW222", name="follower", robot_type="so101_follower", replace=True)
    assert len(registry) == 1
    assert registry.find_by_name("follower").serial == "NEW222"
    assert registry.find_by_name("spare") is None
    assert registry.find_by_serial("OLD111") is None


def test_register_replace_without_conflict_is_plain_add(registry: DeviceRegistry):
    # replace=True on a fresh name behaves like a normal registration.
    device = registry.register(serial="AAA111", name="leader", robot_type="so101_leader", replace=True)
    assert len(registry) == 1
    assert device.name == "leader"


def test_unregister_removes_by_name(registry: DeviceRegistry):
    registry.register(serial="AAA111", name="leader", robot_type="so101_leader")
    assert registry.unregister("leader") is True
    assert len(registry) == 0
    assert registry.unregister("leader") is False  # idempotent on missing


def test_save_and_load_round_trip(tmp_path: Path):
    path = tmp_path / "devices.json"
    reg = DeviceRegistry(path=path)
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.register(serial="BBB222", name="follower", robot_type="so101_follower")
    reg.save()

    assert path.is_file()
    raw = json.loads(path.read_text())
    assert {d["serial"] for d in raw["devices"]} == {"AAA111", "BBB222"}

    reloaded = DeviceRegistry.load(path=path)
    assert len(reloaded) == 2
    assert reloaded.find_by_name("leader").serial == "AAA111"
    assert reloaded.find_by_serial("BBB222").name == "follower"


def test_save_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c" / "devices.json"
    reg = DeviceRegistry(path=nested)
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.save()
    assert nested.is_file()


def test_list_connected_boards_filters_to_so101(two_so101_boards):
    boards = list_connected_boards()
    serials = sorted(b.serial_number for b in boards)
    assert serials == ["AAA111", "BBB222"]
    # Unrelated /dev/ttyS0 and /dev/ttyUSB0 are excluded.
    ports = {b.port for b in boards}
    assert ports == {"/dev/ttyACM0", "/dev/ttyACM1"}


def test_list_connected_boards_skips_missing_serials(fake_comports):
    fake_comports(
        [
            _fake_port("/dev/ttyACM0", SO101_BOARD_VID, SO101_BOARD_PID, None),
            _fake_port("/dev/ttyACM1", SO101_BOARD_VID, SO101_BOARD_PID, ""),
            _fake_port("/dev/ttyACM2", SO101_BOARD_VID, SO101_BOARD_PID, "OK123"),
        ]
    )
    boards = list_connected_boards()
    assert [b.serial_number for b in boards] == ["OK123"]


def test_serial_for_port(two_so101_boards):
    assert serial_for_port("/dev/ttyACM0") == "AAA111"
    assert serial_for_port("/dev/ttyACM1") == "BBB222"
    assert serial_for_port("/dev/nonexistent") is None


def test_resolve_port_returns_matching_port(registry: DeviceRegistry, two_so101_boards):
    registry.register(serial="AAA111", name="leader", robot_type="so101_leader")
    registry.register(serial="BBB222", name="follower", robot_type="so101_follower")
    assert registry.resolve_port("leader") == "/dev/ttyACM0"
    assert registry.resolve_port("follower") == "/dev/ttyACM1"


def test_resolve_port_unknown_name_raises(registry: DeviceRegistry, two_so101_boards):
    with pytest.raises(DeviceNotConnectedError, match="No device named"):
        registry.resolve_port("ghost")


def test_resolve_port_registered_but_unplugged_raises(registry: DeviceRegistry, fake_comports):
    registry.register(serial="AAA111", name="leader", robot_type="so101_leader")
    fake_comports([])  # nothing connected
    with pytest.raises(DeviceNotConnectedError, match="not currently"):
        registry.resolve_port("leader")


def test_resolve_port_survives_port_swap(registry: DeviceRegistry, fake_comports):
    """If the same board reappears on a different ttyACM, resolve_port still finds it."""
    registry.register(serial="AAA111", name="leader", robot_type="so101_leader")
    fake_comports([_fake_port("/dev/ttyACM7", SO101_BOARD_VID, SO101_BOARD_PID, "AAA111")])
    assert registry.resolve_port("leader") == "/dev/ttyACM7"


def test_device_mismatch_error_message_contains_serials():
    err = DeviceMismatchError("leader", expected_serial="AAA111", found_serial="BBB222")
    msg = str(err)
    assert "leader" in msg
    assert "AAA111" in msg
    assert "BBB222" in msg
    assert "register-device" in msg  # actionable


@pytest.fixture
def isolated_registry_path(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("lerobot.utils.device_registry.HF_LEROBOT_CALIBRATION", tmp_path)
    return tmp_path / "devices.json"


def test_resolve_or_verify_port_no_port_no_registration_raises(isolated_registry_path):
    with pytest.raises(ValueError, match="lerobot-register-device --type so101_leader"):
        resolve_or_verify_port("leader", None, register_command_hint="--type so101_leader")


def test_resolve_or_verify_port_no_port_with_registration_auto_resolves(
    isolated_registry_path, two_so101_boards
):
    reg = DeviceRegistry(path=isolated_registry_path)
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.save()
    assert resolve_or_verify_port("leader", None) == "/dev/ttyACM0"


def test_resolve_or_verify_port_explicit_port_unregistered_passthrough(
    isolated_registry_path, two_so101_boards
):
    # No registry entry for 'leader' → explicit port passes through untouched.
    assert resolve_or_verify_port("leader", "/dev/ttyACM0") == "/dev/ttyACM0"


def test_resolve_or_verify_port_explicit_port_matches_registration(isolated_registry_path, two_so101_boards):
    reg = DeviceRegistry(path=isolated_registry_path)
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.save()
    # User correctly passes the port that matches the registered serial.
    assert resolve_or_verify_port("leader", "/dev/ttyACM0") == "/dev/ttyACM0"


def test_resolve_or_verify_port_explicit_port_mismatch_raises(isolated_registry_path, two_so101_boards):
    reg = DeviceRegistry(path=isolated_registry_path)
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.save()
    # User passes a port that points at the OTHER board → cable swap detected.
    with pytest.raises(DeviceMismatchError) as exc_info:
        resolve_or_verify_port("leader", "/dev/ttyACM1")
    assert exc_info.value.expected_serial == "AAA111"
    assert exc_info.value.found_serial == "BBB222"


def test_resolve_or_verify_port_cross_check_unregistered_name_using_others_port(
    isolated_registry_path, two_so101_boards
):
    """Unregistered name + port that belongs to a different registered board → catch the swap."""
    from lerobot.utils.device_registry import BoardClaimedByAnotherNameError

    reg = DeviceRegistry(path=isolated_registry_path)
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.save()
    # 'follower' is NOT registered, but the user is pointing it at the leader's port.
    # Pre-fix this would silently pass; the cross-check should refuse.
    with pytest.raises(BoardClaimedByAnotherNameError) as exc_info:
        resolve_or_verify_port("follower", "/dev/ttyACM0")
    assert exc_info.value.owner_name == "leader"
    assert exc_info.value.board_serial == "AAA111"
    assert exc_info.value.requested_name == "follower"


def test_resolve_or_verify_port_cross_check_no_id_using_registered_port(
    isolated_registry_path, two_so101_boards
):
    """User omits --robot.id but supplies a port that belongs to a registered board → catch the swap."""
    from lerobot.utils.device_registry import BoardClaimedByAnotherNameError

    reg = DeviceRegistry(path=isolated_registry_path)
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.save()
    with pytest.raises(BoardClaimedByAnotherNameError) as exc_info:
        resolve_or_verify_port(None, "/dev/ttyACM0")
    assert exc_info.value.owner_name == "leader"
    assert exc_info.value.requested_name is None


def test_resolve_or_verify_port_cross_check_legacy_passthrough_preserved(
    isolated_registry_path, two_so101_boards
):
    """When neither the name nor the connected board is in the registry, port still passes through."""
    # No registrations at all → fully legacy behavior preserved.
    assert resolve_or_verify_port("follower", "/dev/ttyACM0") == "/dev/ttyACM0"
    assert resolve_or_verify_port(None, "/dev/ttyACM0") == "/dev/ttyACM0"


def test_resolve_or_verify_port_no_port_no_id_gives_clear_error(isolated_registry_path):
    """When the user passes neither --robot.id nor --robot.port, error message should not mention `None`."""
    with pytest.raises(ValueError) as exc_info:
        resolve_or_verify_port(None, None, register_command_hint="--type so101_leader")
    msg = str(exc_info.value)
    assert "None" not in msg.split("`")[0]  # not in the prose part
    assert "--robot.port" in msg
    assert "<friendly_name>" in msg
