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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig
from ..so_leader import SOLeaderConfig


@TeleoperatorConfig.register_subclass("bi_so_leader")
@dataclass
class BiSOLeaderConfig(TeleoperatorConfig):
    """Configuration class for Bi SO Leader teleoperators.

    The two arm configs default to empty ``SOLeaderConfig``s so a bimanual leader
    can be driven by ``--teleop.id=<id>`` alone (each arm's port is then resolved
    from the device registry via ``{id}_left`` / ``{id}_right``). Pass explicit
    ``--teleop.<side>_arm_config.port=...`` to override.
    """

    left_arm_config: SOLeaderConfig = field(default_factory=SOLeaderConfig)
    right_arm_config: SOLeaderConfig = field(default_factory=SOLeaderConfig)
