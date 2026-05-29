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

import pytest

from lerobot.utils.camera_picker import (
    _grid_shape,
    _GuiUnavailableError,
    _should_use_text_mode,
    pick_camera,
)
from lerobot.utils.camera_registry import DiscoveredCamera


def _cam(idx: int, usb_path: str | None = None, capture_node: str | None = None) -> DiscoveredCamera:
    return DiscoveredCamera(
        usb_path=usb_path or f"usb-0:{idx}:1.0",
        vid="0c45",
        pid="6366",
        serial="SN0001",
        model="Innomaker-U20CAM-1080p-S1",
        all_nodes=[capture_node or f"/dev/video{idx}"],
        capture_node=capture_node or f"/dev/video{idx}",
    )


@pytest.fixture
def fake_input(monkeypatch):
    """Queue successive input() responses."""

    def _set(responses):
        it = iter(responses)
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(it))

    return _set


@pytest.fixture
def force_text_mode(monkeypatch):
    monkeypatch.setenv("LEROBOT_HEADLESS_PICKER", "1")


# --- Mode-selection logic --------------------------------------------------


def test_should_use_text_mode_when_env_var_set(monkeypatch):
    monkeypatch.setenv("LEROBOT_HEADLESS_PICKER", "1")
    monkeypatch.setenv("DISPLAY", ":0")
    assert _should_use_text_mode(2) is True


def test_should_use_text_mode_when_too_many_candidates(monkeypatch):
    monkeypatch.delenv("LEROBOT_HEADLESS_PICKER", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    # 10 candidates exceeds the single-digit key limit.
    assert _should_use_text_mode(10) is True


def test_should_use_text_mode_when_no_display(monkeypatch):
    monkeypatch.delenv("LEROBOT_HEADLESS_PICKER", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("lerobot.utils.camera_picker.sys.platform", "linux")
    assert _should_use_text_mode(2) is True


def test_should_use_gui_when_display_set(monkeypatch):
    monkeypatch.delenv("LEROBOT_HEADLESS_PICKER", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr("lerobot.utils.camera_picker.sys.platform", "linux")
    assert _should_use_text_mode(2) is False


# --- Text picker ----------------------------------------------------------


def test_text_picker_returns_chosen_candidate(force_text_mode, fake_input, capsys):
    candidates = [_cam(1), _cam(2), _cam(3)]
    fake_input(["2"])
    result = pick_camera("right_overhead", candidates)
    assert result is candidates[1]
    out = capsys.readouterr().out
    assert "right_overhead" in out
    assert "[2]" in out


def test_text_picker_returns_none_on_quit(force_text_mode, fake_input):
    candidates = [_cam(1), _cam(2)]
    fake_input(["q"])
    assert pick_camera("right_overhead", candidates) is None


def test_text_picker_returns_none_on_empty_input(force_text_mode, fake_input):
    candidates = [_cam(1), _cam(2)]
    fake_input([""])
    assert pick_camera("right_overhead", candidates) is None


def test_text_picker_reprompts_on_invalid_input(force_text_mode, fake_input, capsys):
    candidates = [_cam(1), _cam(2)]
    fake_input(["banana", "9", "0", "1"])  # invalid string, out of range, out of range, valid
    result = pick_camera("right_overhead", candidates)
    assert result is candidates[0]


def test_text_picker_returns_none_for_empty_candidates():
    assert pick_camera("any_name", []) is None


def test_text_picker_shows_no_capture_node_label(force_text_mode, fake_input, capsys):
    no_capture = DiscoveredCamera(
        usb_path="usb-0:7:1.0",
        vid="0c45",
        pid="6366",
        serial="SN0001",
        model="U20CAM",
        all_nodes=["/dev/video9"],
        capture_node=None,
    )
    fake_input(["q"])
    pick_camera("broken", [no_capture])
    assert "<no capture node>" in capsys.readouterr().out


# --- Fallback from GUI to text on cv2 failure -----------------------------


def test_gui_failure_falls_back_to_text(monkeypatch, fake_input, capsys):
    # GUI path is selected (no headless env, display present, candidates ≤ 9)…
    monkeypatch.delenv("LEROBOT_HEADLESS_PICKER", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr("lerobot.utils.camera_picker.sys.platform", "linux")

    # …but _pick_grid raises _GuiUnavailableError (simulating cv2.error from
    # opencv-python-headless), so the picker should fall back to text mode.
    def _boom(name, candidates):
        raise _GuiUnavailableError("simulated no-GUI cv2 build")

    monkeypatch.setattr("lerobot.utils.camera_picker._pick_grid", _boom)
    fake_input(["1"])
    candidates = [_cam(1), _cam(2)]
    result = pick_camera("right_overhead", candidates)
    assert result is candidates[0]
    err = capsys.readouterr().err
    assert "unavailable" in err
    assert "opencv-python" in err  # the hint about which package to install


# --- Grid layout math -----------------------------------------------------


def test_grid_shape_layouts():
    assert _grid_shape(1) == (1, 1)
    assert _grid_shape(2) == (2, 1)
    assert _grid_shape(3) == (2, 2)
    assert _grid_shape(4) == (2, 2)
    assert _grid_shape(5) == (3, 2)
    assert _grid_shape(6) == (3, 2)
    assert _grid_shape(7) == (3, 3)
    assert _grid_shape(9) == (3, 3)
