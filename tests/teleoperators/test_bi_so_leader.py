#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Device-registry resolution for the bimanual SO leader.

``BiSOLeader`` composes two single-arm ``SOLeader`` instances with ``{id}_left`` /
``{id}_right`` ids. These tests prove that each arm's port is independently
resolved from (or verified against) the device registry, exactly like a single
SO leader, so the bimanual setup gets serial-number resolution "for free".
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.device_registry import (
    SO101_BOARD_PID,
    SO101_BOARD_VID,
    DeviceMismatchError,
    DeviceRegistry,
)


def _fake_port(device: str, serial_number: str):
    """A fake SO-101 ``ListPortInfo`` for mocking ``comports()``."""
    return SimpleNamespace(
        device=device, vid=SO101_BOARD_VID, pid=SO101_BOARD_PID, serial_number=serial_number
    )


@pytest.fixture
def registry_env(monkeypatch, tmp_path: Path):
    """Isolate the registry on disk and fake two connected SO-101 boards.

    Returns the ``DeviceRegistry`` so each test decides what to register.
    """
    monkeypatch.setattr("lerobot.utils.device_registry.HF_LEROBOT_CALIBRATION", tmp_path)
    monkeypatch.setattr(
        "lerobot.utils.device_registry._comports",
        lambda: [_fake_port("/dev/ttyACM0", "LEFT111"), _fake_port("/dev/ttyACM1", "RIGHT222")],
    )
    # The composed SOLeader builds a real FeetechMotorsBus; stub it so the test
    # needs no hardware. Port resolution happens before the bus is constructed.
    monkeypatch.setattr(
        "lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus",
        MagicMock(name="FeetechMotorsBus"),
    )
    return DeviceRegistry(path=tmp_path / "devices.json")


def _bi_config(tmp_path: Path, *, id: str | None, left_port=None, right_port=None) -> BiSOLeaderConfig:
    return BiSOLeaderConfig(
        id=id,
        calibration_dir=tmp_path / "calib",
        left_arm_config=SOLeaderConfig(port=left_port),
        right_arm_config=SOLeaderConfig(port=right_port),
    )


def test_bi_leader_resolves_both_ports_from_registry(registry_env, tmp_path: Path):
    registry_env.register(serial="LEFT111", name="bimanual_left", robot_type="so101_leader")
    registry_env.register(serial="RIGHT222", name="bimanual_right", robot_type="so101_leader")
    registry_env.save()

    bi = BiSOLeader(_bi_config(tmp_path, id="bimanual"))

    assert bi.left_arm.config.port == "/dev/ttyACM0"
    assert bi.right_arm.config.port == "/dev/ttyACM1"


def test_bi_leader_explicit_ports_pass_through_when_unregistered(registry_env, tmp_path: Path):
    # Nothing registered → user-supplied ports are honored untouched (legacy behavior).
    bi = BiSOLeader(_bi_config(tmp_path, id="bimanual", left_port="/dev/ttyACM0", right_port="/dev/ttyACM1"))
    assert bi.left_arm.config.port == "/dev/ttyACM0"
    assert bi.right_arm.config.port == "/dev/ttyACM1"


def test_bi_leader_swapped_cable_on_one_arm_raises(registry_env, tmp_path: Path):
    registry_env.register(serial="LEFT111", name="bimanual_left", robot_type="so101_leader")
    registry_env.register(serial="RIGHT222", name="bimanual_right", robot_type="so101_leader")
    registry_env.save()

    # Left arm is pointed at the RIGHT board's port → cable swap must be caught.
    with pytest.raises(DeviceMismatchError) as exc_info:
        BiSOLeader(_bi_config(tmp_path, id="bimanual", left_port="/dev/ttyACM1"))
    assert exc_info.value.expected_serial == "LEFT111"
    assert exc_info.value.found_serial == "RIGHT222"


def test_bi_leader_no_id_no_port_raises_clear_error(registry_env, tmp_path: Path):
    # No id (so no registry name) and no ports → actionable error, no bare "None".
    with pytest.raises(ValueError) as exc_info:
        BiSOLeader(_bi_config(tmp_path, id=None))
    msg = str(exc_info.value)
    assert "--robot.port" in msg
    assert "lerobot-register-device" in msg


def test_bi_leader_registered_left_only_still_needs_right(registry_env, tmp_path: Path):
    # Only the left arm is registered; the right has neither registration nor port.
    registry_env.register(serial="LEFT111", name="bimanual_left", robot_type="so101_leader")
    registry_env.save()
    with pytest.raises(ValueError, match="bimanual_right"):
        BiSOLeader(_bi_config(tmp_path, id="bimanual"))


def test_bi_leader_config_parses_from_id_alone():
    """Fix #1: the bimanual config builds from --id alone (no per-arm dummies)."""
    import draccus

    cfg = draccus.parse(BiSOLeaderConfig, args=["--id=foo"])
    assert cfg.left_arm_config.port is None
    assert cfg.right_arm_config.port is None


def test_bi_leader_disconnect_tears_down_both_even_if_one_fails(registry_env, tmp_path: Path):
    """Fix #2: if one arm's disconnect raises, the other arm is still disconnected."""
    registry_env.register(serial="LEFT111", name="bimanual_left", robot_type="so101_leader")
    registry_env.register(serial="RIGHT222", name="bimanual_right", robot_type="so101_leader")
    registry_env.save()
    bi = BiSOLeader(_bi_config(tmp_path, id="bimanual"))

    bi.left_arm = MagicMock()
    bi.right_arm = MagicMock()
    bi.left_arm.disconnect.side_effect = RuntimeError("left boom")

    with pytest.raises(RuntimeError, match="left boom"):
        bi.disconnect()
    bi.right_arm.disconnect.assert_called_once()
