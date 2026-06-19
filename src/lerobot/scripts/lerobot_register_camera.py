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
Pair a connected camera to a friendly name.

Discovers both **UVC/V4L2 webcams** and **Intel RealSense** cameras. RealSense
(and any camera with a unique serial) is bound by serial — **port-independent**,
so replugging it never needs a re-pair. Cheap webcams that share a serial (e.g.
Innomaker U20CAMs all report ``SN0001``) are disambiguated with an interactive
picker — a tiled grid of live previews if a display is available, or a numbered
text list otherwise. The binding is saved to
``$HF_LEROBOT_CALIBRATION/cameras.json`` and subsequent commands refer to the
camera by name:

  - UVC:       ``--robot.cameras="{ cam: {type: opencv, id: my_cam, ...}}"``
  - RealSense: ``--robot.cameras="{ cam: {type: intelrealsense, id: my_cam, ...}}"``

Example:

```shell
lerobot-register-camera --name right_overhead
lerobot-register-camera --name top_depth       # an Intel RealSense, if connected
lerobot-register-camera --name wrist_cam --headless    # force text mode
lerobot-register-camera --name workspace --force       # skip re-pair confirmation
```
"""

from __future__ import annotations

import argparse
import os
import sys

from lerobot.utils.camera_picker import assign_names, pick_camera
from lerobot.utils.camera_registry import (
    KIND_REALSENSE,
    CameraRegistry,
    DiscoveredCamera,
    discover_cameras,
)


def _format_camera_row(idx: int, cam: DiscoveredCamera, registry: CameraRegistry) -> str:
    existing = registry.find_by_camera(cam)
    suffix = "(unregistered)" if existing is None else f'(registered as "{existing.name}")'
    if cam.kind == KIND_REALSENSE:
        return f"  [{idx}] [RealSense] {cam.model}  serial={cam.serial}  {suffix}"
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
        cameras = discover_cameras()
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
        print("No cameras detected. Plug one in and re-run.", file=sys.stderr)
        return 1

    registry = CameraRegistry.load()

    print(f"Found {len(cameras)} camera(s):")
    for i, cam in enumerate(cameras, start=1):
        print(_format_camera_row(i, cam, registry))
    print()

    chosen = pick_camera(name, cameras)
    if chosen is None:
        print("Cancelled. Registry unchanged.", file=sys.stderr)
        return 1

    # UVC cameras must expose a V4L2 capture node; RealSense legitimately has none
    # (it's driven through the SDK and identified by serial).
    if chosen.kind != KIND_REALSENSE and chosen.capture_node is None:
        print(
            f"Selected camera at {chosen.usb_path} has no V4L2 capture-capable node. "
            "Pick a different camera or check the device.",
            file=sys.stderr,
        )
        return 1

    # If this physical camera is already registered under a different name,
    # confirm before overwriting.
    existing = registry.find_by_camera(chosen)
    if existing is not None and existing.name != name:
        if not force and not _prompt_yes_no(
            f'This camera is already registered as "{existing.name}". Re-pair it as "{name}"?',
            default=False,
        ):
            print("Cancelled. Registry unchanged.", file=sys.stderr)
            return 1
        # Drop the camera's stale old name so re-pairing doesn't leave a duplicate
        # entry for the same physical camera (applies under --force too).
        registry.unregister(existing.name)

    # replace=True: re-pairing also overwrites any stale binding still holding `name`
    # (e.g. a previously-registered camera that's since been unplugged/moved), so the
    # user never has to manually unregister the old entry first.
    registry.register(chosen, name, replace=True)

    registry.save()
    if chosen.kind == KIND_REALSENSE:
        print(f'\n✓ Registered "{name}" → {chosen.model} (RealSense, serial {chosen.serial})')
        print("  identified by serial — port-independent (no re-pair on replug).")
    else:
        print(f'\n✓ Registered "{name}" → {chosen.model} at {chosen.usb_path}')
        print(f"  capture node: {chosen.capture_node}")
    print(f"  saved to {registry.path}")
    return 0


def register_all(*, headless: bool = False) -> int:
    """Batch flow: show a snapshot of every camera and name them all in one session."""
    if headless:
        os.environ["LEROBOT_HEADLESS_PICKER"] = "1"

    try:
        cameras = discover_cameras()
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
        print("No cameras detected. Plug one in and re-run.", file=sys.stderr)
        return 1

    registry = CameraRegistry.load()
    print(f"Found {len(cameras)} camera(s):")
    for i, cam in enumerate(cameras, start=1):
        print(_format_camera_row(i, cam, registry))

    assignments = assign_names(cameras)
    if not assignments:
        print("No names entered. Registry unchanged.", file=sys.stderr)
        return 1

    # Reject duplicate names within a single batch (save nothing — avoid a half-applied mess).
    names = [name for _, name in assignments]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        print(
            f"Duplicate name(s) in this batch: {', '.join(duplicates)}. "
            "Each camera needs a unique name. Nothing was saved.",
            file=sys.stderr,
        )
        return 1

    # Naming a camera is the user's explicit re-pair consent, so register without
    # per-camera prompts: drop any stale binding the chosen camera holds, then save.
    for cam, name in assignments:
        existing = registry.find_by_camera(cam)
        if existing is not None and existing.name != name:
            registry.unregister(existing.name)
        registry.register(cam, name, replace=True)
    registry.save()

    print("\n✓ Registered:")
    for cam, name in assignments:
        key = f"serial {cam.serial}" if cam.kind == KIND_REALSENSE else cam.usb_path
        print(f"  {name}  ({cam.kind})  → {key}")
    print(f"  saved to {registry.path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lerobot-register-camera",
        description="Pair connected cameras (UVC webcams + Intel RealSense) to friendly names.",
    )
    parser.add_argument(
        "--name",
        help='Friendly name for a single camera (e.g. "right_overhead"). Omit when using --all.',
    )
    parser.add_argument(
        "--all",
        dest="register_all",
        action="store_true",
        help="Batch mode: show a snapshot of every camera and name them all in one session.",
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

    if args.register_all and args.name:
        parser.error("pass either --name <name> (single) or --all (batch), not both.")
    if args.register_all:
        sys.exit(register_all(headless=args.headless))
    if not args.name:
        parser.error("pass --name <name> to register one camera, or --all to register them all.")
    sys.exit(register_camera(args.name, headless=args.headless, force=args.force))


if __name__ == "__main__":
    main()
