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
Pair an SO-101 controller board to a friendly robot name.

Each board has a factory-burned USB serial number that persists across
reboots and replugs. Pairing the serial to a name once means subsequent
``lerobot-record``, ``lerobot-calibrate`` and ``lerobot-teleoperate``
invocations no longer need a ``--robot.port=/dev/ttyACM0`` flag.

Example:

```shell
lerobot-register-device --type so101_leader --name leader
lerobot-register-device --type so101_follower --name follower
```
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS, TELEOPERATORS
from lerobot.utils.device_registry import (
    ConnectedBoard,
    DeviceNameConflictError,
    DeviceRegistry,
    list_connected_boards,
)

# CLI type string  →  (calibration top-level dir, robot-class .name attribute, kind)
# The CLI type string ("so101_leader") is the @register_subclass key used in calibrate /
# record / teleoperate. The class-name string ("so_leader") is what the calibration
# directory is keyed by — shared between so100 and so101 variants.
_SUPPORTED_TYPES: dict[str, tuple[str, str, str]] = {
    "so101_leader": (TELEOPERATORS, "so_leader", "teleop"),
    "so101_follower": (ROBOTS, "so_follower", "robot"),
}


def _calibration_path(robot_type: str, name: str) -> Path:
    top_dir, class_name, _kind = _SUPPORTED_TYPES[robot_type]
    return HF_LEROBOT_CALIBRATION / top_dir / class_name / f"{name}.json"


def _format_board_line(board: ConnectedBoard, registry: DeviceRegistry) -> str:
    existing = registry.find_by_serial(board.serial_number)
    if existing is None:
        suffix = "(unregistered)"
    else:
        suffix = f'(registered as "{existing.name}" / {existing.robot_type})'
    return f"  {board.port}  serial {board.serial_number}  {suffix}"


def _prompt_yes_no(question: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {hint} ").strip().lower()
        if answer == "":
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _choose_board(boards: list[ConnectedBoard], registry: DeviceRegistry, name: str) -> ConnectedBoard | None:
    """Ask the user which connected board to pair with ``name``.

    Returns ``None`` if the user declines or there's nothing to pair.
    """
    if not boards:
        print("No SO-101 boards detected. Plug one in and re-run.")
        return None

    print(f"Found {len(boards)} SO-101 board(s):")
    for idx, board in enumerate(boards, start=1):
        print(f"  [{idx}] {_format_board_line(board, registry).lstrip()}")
    print()

    unregistered = [b for b in boards if registry.find_by_serial(b.serial_number) is None]

    # Re-pair flow: name is taken by a board that's connected.
    existing_for_name = registry.find_by_name(name)
    if existing_for_name is not None:
        connected_for_name = next((b for b in boards if b.serial_number == existing_for_name.serial), None)
        if connected_for_name is not None:
            print(
                f'Name "{name}" is already registered to board {existing_for_name.serial} '
                f"({connected_for_name.port}). Nothing to do."
            )
            return None
        print(
            f'Name "{name}" is currently mapped to board {existing_for_name.serial}, which is not connected.'
        )
        if not _prompt_yes_no(f'Re-pair "{name}" to a different board?', default=False):
            return None

    if len(unregistered) == 1 and existing_for_name is None:
        board = unregistered[0]
        if _prompt_yes_no(f'Register {board.port} (serial {board.serial_number}) as "{name}"?'):
            return board
        return None

    candidates = unregistered if unregistered else boards
    while True:
        raw = input(
            f"Which board should be paired with \"{name}\"? [1-{len(boards)}, or 'q' to cancel] "
        ).strip()
        if raw.lower() in {"q", "quit", ""}:
            return None
        try:
            choice = int(raw)
        except ValueError:
            print("Enter a number or 'q'.")
            continue
        if not 1 <= choice <= len(boards):
            print(f"Pick a number between 1 and {len(boards)}.")
            continue
        picked = boards[choice - 1]
        existing = registry.find_by_serial(picked.serial_number)
        if (
            existing is not None
            and picked not in candidates
            and not _prompt_yes_no(
                f'Board {picked.serial_number} is already registered as "{existing.name}". Re-pair it?',
                default=False,
            )
        ):
            continue
        return picked


def _maybe_launch_calibration(robot_type: str, name: str) -> None:
    if _calibration_path(robot_type, name).is_file():
        return
    print(f'\nNo calibration file found for "{name}".')
    if not _prompt_yes_no("Run calibration now?", default=True):
        print(
            f"Skipped. When ready, run:\n"
            f"  lerobot-calibrate --{_SUPPORTED_TYPES[robot_type][2]}.type={robot_type} "
            f"--{_SUPPORTED_TYPES[robot_type][2]}.id={name}"
        )
        return
    kind = _SUPPORTED_TYPES[robot_type][2]  # "robot" or "teleop"
    cmd = [
        "lerobot-calibrate",
        f"--{kind}.type={robot_type}",
        f"--{kind}.id={name}",
    ]
    print(f"\n→ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=False)


def register_device(robot_type: str, name: str, *, run_calibration: bool = True) -> int:
    """Pair a connected board to ``name``. Returns process exit code."""
    if robot_type not in _SUPPORTED_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_TYPES))
        print(f"Unsupported type {robot_type!r}. Supported: {supported}", file=sys.stderr)
        return 2

    try:
        boards = list_connected_boards()
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except PermissionError as exc:
        print(f"Permission denied reading USB devices: {exc}", file=sys.stderr)
        if sys.platform.startswith("linux"):
            print(
                "On Linux, ensure your user is in the 'dialout' group:\n"
                "  sudo usermod -aG dialout $USER && newgrp dialout",
                file=sys.stderr,
            )
        return 1

    registry = DeviceRegistry.load()
    board = _choose_board(boards, registry, name)
    if board is None:
        return 1

    try:
        registry.register(serial=board.serial_number, name=name, robot_type=robot_type)
    except DeviceNameConflictError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    registry.save()
    print(f'\n✓ Registered "{name}" → board {board.serial_number} ({board.port}).')
    print(f"  Saved to {registry.path}")

    if run_calibration:
        _maybe_launch_calibration(robot_type, name)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lerobot-register-device",
        description="Pair a connected SO-101 controller board to a friendly robot name.",
    )
    parser.add_argument(
        "--type",
        dest="robot_type",
        required=True,
        choices=sorted(_SUPPORTED_TYPES),
        help="Robot type to register (e.g. so101_leader, so101_follower).",
    )
    parser.add_argument(
        "--name",
        required=True,
        help='Friendly name to use thereafter (e.g. "leader", "follower", "left_arm").',
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip the offer to launch lerobot-calibrate after registration.",
    )
    args = parser.parse_args()
    sys.exit(register_device(args.robot_type, args.name, run_calibration=not args.no_calibrate))


if __name__ == "__main__":
    main()
