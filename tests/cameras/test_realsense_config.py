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

"""RealSenseCameraConfig validation — no pyrealsense2 SDK needed (pure dataclass)."""

import pytest

# The config module imports only from ..configs, so this runs without pyrealsense2.
from lerobot.cameras.realsense import RealSenseCameraConfig


def test_id_only_is_valid():
    cfg = RealSenseCameraConfig(id="top_depth")
    assert cfg.id == "top_depth"
    assert cfg.serial_number_or_name is None


def test_serial_only_is_valid():
    cfg = RealSenseCameraConfig(serial_number_or_name="123456789")
    assert cfg.serial_number_or_name == "123456789"
    assert cfg.id is None


def test_neither_id_nor_serial_raises():
    with pytest.raises(ValueError, match="exactly one of `id`"):
        RealSenseCameraConfig()


def test_both_id_and_serial_raises():
    with pytest.raises(ValueError, match="exactly one of `id`"):
        RealSenseCameraConfig(id="top_depth", serial_number_or_name="123456789")
