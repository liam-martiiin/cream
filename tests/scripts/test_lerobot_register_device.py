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
from types import SimpleNamespace

import pytest

from lerobot.scripts import lerobot_register_device as cmd
from lerobot.utils.device_registry import (
    SO101_BOARD_PID,
    SO101_BOARD_VID,
    DeviceRegistry,
)


def _fake_port(device: str, serial_number: str):
    return SimpleNamespace(
        device=device, vid=SO101_BOARD_VID, pid=SO101_BOARD_PID, serial_number=serial_number
    )


@pytest.fixture
def two_boards(monkeypatch):
    monkeypatch.setattr(
        "lerobot.utils.device_registry._comports",
        lambda: [
            _fake_port("/dev/ttyACM0", "AAA111"),
            _fake_port("/dev/ttyACM1", "BBB222"),
        ],
    )


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path: Path):
    """Point HF_LEROBOT_CALIBRATION at a temp dir so save() doesn't touch the real cache."""
    path = tmp_path / "devices.json"
    monkeypatch.setattr("lerobot.utils.device_registry.HF_LEROBOT_CALIBRATION", tmp_path)
    # Also patch the module-level constant the script imports for calibration paths.
    monkeypatch.setattr("lerobot.scripts.lerobot_register_device.HF_LEROBOT_CALIBRATION", tmp_path)
    return path


@pytest.fixture
def fake_input(monkeypatch):
    """Queue successive input() responses."""

    def _set(responses):
        it = iter(responses)
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(it))

    return _set


@pytest.fixture
def no_calibration_launch(monkeypatch):
    """Stub out subprocess.run so tests never actually launch lerobot-calibrate."""
    calls = []
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_register_device.subprocess.run",
        lambda cmd, check=False: calls.append(cmd) or SimpleNamespace(returncode=0),
    )
    return calls


def test_rejects_unsupported_type(capsys):
    rc = cmd.register_device("nonsense_type", "leader")
    assert rc == 2
    assert "Unsupported type" in capsys.readouterr().err


def test_no_boards_returns_failure(monkeypatch, isolated_registry, fake_input):
    monkeypatch.setattr("lerobot.utils.device_registry._comports", lambda: [])
    fake_input([])  # no input needed
    rc = cmd.register_device("so101_leader", "leader", run_calibration=False)
    assert rc == 1


def test_single_unregistered_board_pairs_with_yes(
    two_boards, isolated_registry, fake_input, no_calibration_launch, capsys
):
    # Only one unregistered board (we pre-register the other).
    reg = DeviceRegistry.load()
    reg.register(serial="AAA111", name="follower", robot_type="so101_follower")
    reg.save()

    fake_input(["y", "n"])  # confirm pairing, then decline calibration
    rc = cmd.register_device("so101_leader", "leader")

    assert rc == 0
    reloaded = DeviceRegistry.load()
    assert reloaded.find_by_name("leader").serial == "BBB222"
    out = capsys.readouterr().out
    assert "Registered" in out


def test_user_decline_does_not_save(two_boards, isolated_registry, fake_input, no_calibration_launch):
    # Both boards unregistered → script asks for an index. 'q' aborts.
    fake_input(["q"])
    rc = cmd.register_device("so101_leader", "leader")
    assert rc == 1
    assert DeviceRegistry.load().find_by_name("leader") is None


def test_multi_unregistered_prompts_for_index(
    two_boards, isolated_registry, fake_input, no_calibration_launch
):
    fake_input(["2", "n"])  # pick board #2, decline calibration
    rc = cmd.register_device("so101_leader", "leader")
    assert rc == 0
    assert DeviceRegistry.load().find_by_name("leader").serial == "BBB222"


def test_quitting_index_prompt_aborts(two_boards, isolated_registry, fake_input):
    fake_input(["q"])
    rc = cmd.register_device("so101_leader", "leader")
    assert rc == 1
    assert DeviceRegistry.load().find_by_name("leader") is None


def test_name_already_registered_to_connected_board_is_noop(two_boards, isolated_registry, fake_input):
    reg = DeviceRegistry.load()
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.save()
    fake_input([])  # script shouldn't prompt
    rc = cmd.register_device("so101_leader", "leader", run_calibration=False)
    # Already paired correctly → returns 1 (nothing to do) but doesn't corrupt registry.
    assert rc == 1
    assert DeviceRegistry.load().find_by_name("leader").serial == "AAA111"


def test_launches_calibration_when_no_calibration_file(
    two_boards, isolated_registry, fake_input, no_calibration_launch, tmp_path: Path
):
    fake_input(["1", "y"])  # pick board 1, accept calibration
    rc = cmd.register_device("so101_follower", "follower")
    assert rc == 0
    assert no_calibration_launch == [
        ["lerobot-calibrate", "--robot.type=so101_follower", "--robot.id=follower"]
    ]


def test_skips_calibration_when_file_exists(
    two_boards, isolated_registry, fake_input, no_calibration_launch, tmp_path: Path
):
    cal = tmp_path / "robots" / "so_follower" / "follower.json"
    cal.parent.mkdir(parents=True)
    cal.write_text("{}")
    fake_input(["1"])  # pick board 1, no calibration prompt should appear
    rc = cmd.register_device("so101_follower", "follower")
    assert rc == 0
    assert no_calibration_launch == []  # no subprocess invocation


def test_repair_name_to_new_board_when_old_is_disconnected(
    monkeypatch, isolated_registry, fake_input, no_calibration_launch
):
    """Regression: a name bound to a now-disconnected board, with only a new board plugged in.

    Previously the re-pair flow asked for confirmation, accepted the new board, then
    failed with DeviceNameConflictError because the name still pointed at the old serial.
    """
    monkeypatch.setattr(
        "lerobot.utils.device_registry._comports",
        lambda: [_fake_port("/dev/ttyACM2", "NEW333")],
    )
    reg = DeviceRegistry.load()
    reg.register(serial="OLD999", name="follower", robot_type="so101_follower")
    reg.save()

    fake_input(["y", "1"])  # confirm re-pair, then pick the single connected board
    rc = cmd.register_device("so101_follower", "follower", run_calibration=False)

    assert rc == 0
    reloaded = DeviceRegistry.load()
    assert reloaded.find_by_name("follower").serial == "NEW333"
    assert reloaded.find_by_serial("OLD999") is None
    assert len(reloaded) == 1  # no stale duplicate left behind


def test_repair_declined_leaves_registry_untouched(
    monkeypatch, isolated_registry, fake_input, no_calibration_launch
):
    monkeypatch.setattr(
        "lerobot.utils.device_registry._comports",
        lambda: [_fake_port("/dev/ttyACM2", "NEW333")],
    )
    reg = DeviceRegistry.load()
    reg.register(serial="OLD999", name="follower", robot_type="so101_follower")
    reg.save()

    fake_input(["n"])  # decline the re-pair prompt
    rc = cmd.register_device("so101_follower", "follower", run_calibration=False)

    assert rc == 1
    assert DeviceRegistry.load().find_by_name("follower").serial == "OLD999"


def test_unregister_removes_entry(isolated_registry, capsys):
    reg = DeviceRegistry.load()
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.save()

    rc = cmd.unregister_device("leader")

    assert rc == 0
    assert DeviceRegistry.load().find_by_name("leader") is None
    assert "Unregistered" in capsys.readouterr().out


def test_unregister_missing_name_returns_1(isolated_registry, capsys):
    rc = cmd.unregister_device("ghost")
    assert rc == 1
    assert "Nothing to do" in capsys.readouterr().err


def test_main_unregister_dispatch(isolated_registry, monkeypatch):
    reg = DeviceRegistry.load()
    reg.register(serial="AAA111", name="leader", robot_type="so101_leader")
    reg.save()
    monkeypatch.setattr("sys.argv", ["lerobot-register-device", "--unregister", "--name", "leader"])
    with pytest.raises(SystemExit) as exc_info:
        cmd.main()
    assert exc_info.value.code == 0
    assert DeviceRegistry.load().find_by_name("leader") is None


def test_main_register_requires_type(isolated_registry, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["lerobot-register-device", "--name", "leader"])
    with pytest.raises(SystemExit) as exc_info:
        cmd.main()
    assert exc_info.value.code != 0  # argparse parser.error exits 2
    assert "--type is required" in capsys.readouterr().err


def test_calibration_path_uses_class_dir_not_cli_type(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("lerobot.scripts.lerobot_register_device.HF_LEROBOT_CALIBRATION", tmp_path)
    leader = cmd._calibration_path("so101_leader", "leader")
    follower = cmd._calibration_path("so101_follower", "follower")
    assert leader == tmp_path / "teleoperators" / "so_leader" / "leader.json"
    assert follower == tmp_path / "robots" / "so_follower" / "follower.json"
