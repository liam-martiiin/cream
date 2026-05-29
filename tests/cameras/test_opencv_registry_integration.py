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
Tests for the camera-registry wiring on ``OpenCVCameraConfig`` / ``OpenCVCamera``:

- Config validation that ``id`` and ``index_or_path`` are mutually exclusive
  and exactly one must be set.
- ``OpenCVCamera.__init__`` resolves a friendly ``id`` to a concrete
  ``/dev/video*`` path via the registry, and skips that path when the user
  passes ``index_or_path`` directly.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# --- Config validation -----------------------------------------------------


def test_config_accepts_index_or_path():
    cfg = OpenCVCameraConfig(index_or_path=0)
    assert cfg.index_or_path == 0
    assert cfg.id is None


def test_config_accepts_dev_path():
    cfg = OpenCVCameraConfig(index_or_path=Path("/dev/video0"))
    assert cfg.index_or_path == Path("/dev/video0")


def test_config_accepts_id():
    cfg = OpenCVCameraConfig(id="right_overhead")
    assert cfg.id == "right_overhead"
    assert cfg.index_or_path is None


def test_config_rejects_neither_set():
    with pytest.raises(ValueError, match="exactly one"):
        OpenCVCameraConfig()


def test_config_rejects_both_set():
    with pytest.raises(ValueError, match="exactly one"):
        OpenCVCameraConfig(index_or_path=0, id="right_overhead")


# --- Camera __init__ registry resolve --------------------------------------


def test_camera_with_id_resolves_via_registry(monkeypatch):
    fake_registry = MagicMock()
    fake_registry.resolve.return_value = "/dev/video7"
    monkeypatch.setattr(
        "lerobot.utils.camera_registry.CameraRegistry.load",
        classmethod(lambda cls: fake_registry),
    )
    cfg = OpenCVCameraConfig(id="right_overhead")
    camera = OpenCVCamera(cfg)
    assert camera.index_or_path == "/dev/video7"
    fake_registry.resolve.assert_called_once()
    args, kwargs = fake_registry.resolve.call_args
    assert args == ("right_overhead",)
    # The picker arg should be wired up so resolve() can self-heal interactively.
    assert "picker" in kwargs


def test_camera_with_index_or_path_skips_registry(monkeypatch):
    # If the registry were touched, the test would blow up on the resolve call.
    fake_registry = MagicMock()
    fake_registry.resolve.side_effect = AssertionError("registry should not be touched")
    monkeypatch.setattr(
        "lerobot.utils.camera_registry.CameraRegistry.load",
        classmethod(lambda cls: fake_registry),
    )
    cfg = OpenCVCameraConfig(index_or_path=0)
    camera = OpenCVCamera(cfg)
    assert camera.index_or_path == 0
    fake_registry.resolve.assert_not_called()


def test_camera_propagates_registry_error(monkeypatch):
    from lerobot.utils.camera_registry import CameraNotConnectedError

    fake_registry = MagicMock()
    fake_registry.resolve.side_effect = CameraNotConnectedError("missing")
    monkeypatch.setattr(
        "lerobot.utils.camera_registry.CameraRegistry.load",
        classmethod(lambda cls: fake_registry),
    )
    cfg = OpenCVCameraConfig(id="right_overhead")
    with pytest.raises(CameraNotConnectedError, match="missing"):
        OpenCVCamera(cfg)
