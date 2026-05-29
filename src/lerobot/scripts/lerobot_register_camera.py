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
Pair a connected UVC camera to a friendly name.

Where cheap webcams share USB serial numbers (e.g. Innomaker U20CAMs all
report ``SN0001``), this script disambiguates with an interactive picker —
a tiled grid of live previews if a display is available, or a numbered text
list otherwise. After picking, the binding is saved to
``$HF_LEROBOT_CALIBRATION/cameras.json`` and subsequent commands can refer
to the camera by name (``--robot.cameras="{ cam: {type: opencv, id: my_cam, ...}}"``).

Example:

```shell
lerobot-register-camera --name right_overhead
lerobot-register-camera --name wrist_cam --headless    # force text mode
lerobot-register-camera --name workspace --force       # skip re-pair confirmation
```
"""

from __future__ import annotations

import argparse
import os
import sys

from lerobot.utils.camera_picker import pick_camera
from lerobot.utils.camera_registry import (
    CameraNameConflictError,
    CameraRegistry,
    DiscoveredCamera,
    discover_uvc_cameras,
)


def _format_camera_row(idx: int, cam: DiscoveredCamera, registry: CameraRegistry) -> str:
    existing = next(
        (
            r
            for r in registry
            if r.vid == cam.vid and r.pid == cam.pid and r.serial == cam.serial and r.usb_path == cam.usb_path
        ),
        None,
    )
    suffix = "(unregistered)" if existing is None else f'(registered as "{existing.name}")'
    capture = cam.capture_node if cam.capture_node else "<no capture node>"
    return f"  [{idx}] {cam.model}  usb_path={cam.usb_path}  capture={capture}  {suffix}"


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


def register_camera(name: str, *, headless: bool = False, force: bool = False) -> int:
    """Run the registration flow for a single name. Returns process exit code."""
    if headless:
        # `pick_camera()` already honors this env var; setting it from here means
        # the user gets the text picker without any of the auto-detection logic.
        os.environ["LEROBOT_HEADLESS_PICKER"] = "1"

    try:
        cameras = discover_uvc_cameras()
    except PermissionError as exc:
        print(f"Permission denied reading video devices: {exc}", file=sys.stderr)
        if sys.platform.startswith("linux"):
            print(
                "On Linux, ensure your user is in the 'video' group:\n"
                "  sudo usermod -aG video $USER && newgrp video",
                file=sys.stderr,
            )
        return 1

    if not cameras:
        print("No UVC cameras detected. Plug one in and re-run.", file=sys.stderr)
        return 1

    registry = CameraRegistry.load()

    print(f"Found {len(cameras)} UVC camera(s):")
    for i, cam in enumerate(cameras, start=1):
        print(_format_camera_row(i, cam, registry))
    print()

    chosen = pick_camera(name, cameras)
    if chosen is None:
        print("Cancelled. Registry unchanged.", file=sys.stderr)
        return 1

    if chosen.capture_node is None:
        print(
            f"Selected camera at {chosen.usb_path} has no V4L2 capture-capable node. "
            "Pick a different camera or check the device.",
            file=sys.stderr,
        )
        return 1

    # If this physical camera is already registered under a different name,
    # confirm before overwriting.
    existing = next(
        (
            r
            for r in registry
            if r.vid == chosen.vid
            and r.pid == chosen.pid
            and r.serial == chosen.serial
            and r.usb_path == chosen.usb_path
        ),
        None,
    )
    if existing is not None and existing.name != name and not force:
        if not _prompt_yes_no(
            f'This camera is already registered as "{existing.name}". Re-pair it as "{name}"?',
            default=False,
        ):
            print("Cancelled. Registry unchanged.", file=sys.stderr)
            return 1
        registry.unregister(existing.name)

    try:
        registry.register(chosen, name)
    except CameraNameConflictError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    registry.save()
    print(f'\n✓ Registered "{name}" → {chosen.model} at {chosen.usb_path}')
    print(f"  capture node: {chosen.capture_node}")
    print(f"  saved to {registry.path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lerobot-register-camera",
        description="Pair a connected UVC camera to a friendly name.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help='Friendly name to use thereafter (e.g. "right_overhead", "wrist_cam").',
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force the text-mode picker even if a display is available.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the re-pair confirmation when the chosen camera is already registered.",
    )
    args = parser.parse_args()
    sys.exit(register_camera(args.name, headless=args.headless, force=args.force))


if __name__ == "__main__":
    main()
