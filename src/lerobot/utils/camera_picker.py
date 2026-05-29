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
Interactive camera picker — used by `lerobot-register-camera` and by the
self-healing path in :pyfunc:`CameraRegistry.resolve` when an ambiguous match
needs the user to disambiguate.

Two interaction modes:

- **Grid view** (the default when a usable display is detected): a single
  window shows a tiled live preview of every candidate camera, with a large
  number overlay on each tile. The user presses ``1``..``N`` to bind.
- **Text mode** (used over SSH, when ``LEROBOT_HEADLESS_PICKER=1`` is set,
  or when ``cv2.imshow`` raises because the user has ``opencv-python-headless``
  rather than the full ``opencv-python`` package): a numbered list is printed
  and the user types the index.

The grid view caps at 9 candidates (single-digit keys). Larger fleets fall
through to text mode where you can type multi-digit indices.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.utils.camera_registry import DiscoveredCamera

# Tile dimensions for the grid preview. Kept low so multiple cameras can
# stream simultaneously without saturating USB 2.0 bandwidth — the picker
# does NOT need to match the eventual capture resolution.
_TILE_WIDTH = 320
_TILE_HEIGHT = 240
_MAX_GRID_CANDIDATES = 9  # bound by single-digit keypresses


def pick_camera(name: str, candidates: list[DiscoveredCamera]) -> DiscoveredCamera | None:
    """Ask the user to pick one of ``candidates`` to bind to friendly name ``name``.

    Returns the chosen ``DiscoveredCamera``, or ``None`` if the user cancelled.

    Routes between the grid view and text mode automatically. Callers don't
    need to know which mode was used.
    """
    if not candidates:
        return None
    if _should_use_text_mode(len(candidates)):
        return _pick_text(name, candidates)
    try:
        return _pick_grid(name, candidates)
    except _GuiUnavailableError as exc:
        print(
            f"Camera preview window unavailable ({exc}). "
            "Install opencv-python (instead of opencv-python-headless) for a grid view. "
            "Falling back to text picker.",
            file=sys.stderr,
        )
        return _pick_text(name, candidates)


def _should_use_text_mode(num_candidates: int) -> bool:
    if os.environ.get("LEROBOT_HEADLESS_PICKER") == "1":
        return True
    if num_candidates > _MAX_GRID_CANDIDATES:
        return True
    if sys.platform == "linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return True
    return False


def _pick_text(name: str, candidates: list[DiscoveredCamera]) -> DiscoveredCamera | None:
    print(f'\nPick a camera to register as "{name}":')
    for i, c in enumerate(candidates, start=1):
        capture = c.capture_node if c.capture_node else "<no capture node>"
        print(f"  [{i}] {c.model}  usb_path={c.usb_path}  capture={capture}")
    while True:
        raw = input(f"Enter 1-{len(candidates)} (or q to cancel): ").strip().lower()
        if raw in {"q", "quit", ""}:
            return None
        try:
            idx = int(raw)
        except ValueError:
            print("Enter a number or 'q'.")
            continue
        if not 1 <= idx <= len(candidates):
            print(f"Pick a number between 1 and {len(candidates)}.")
            continue
        return candidates[idx - 1]


# ---------------------------------------------------------------------------
# Grid view (GUI). Imports cv2/numpy lazily so the text-only path doesn't
# require them, and so import errors only surface when the picker is used.
# ---------------------------------------------------------------------------


class _GuiUnavailableError(RuntimeError):
    """Raised by the GUI path when cv2 GUI bindings aren't available."""


def _pick_grid(name: str, candidates: list[DiscoveredCamera]) -> DiscoveredCamera | None:
    import cv2  # noqa: F401  - imported lazily

    window_title = f"Pick camera for {name!r} — press 1-{len(candidates)} or q to cancel"

    captures: list[object | None] = []
    for cam in candidates:
        captures.append(_open_capture(cam))

    try:
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise _GuiUnavailableError(str(exc).splitlines()[-1] if str(exc) else "no GUI") from exc

        cols, rows = _grid_shape(len(candidates))
        digit_keys = {ord(str(i + 1)): i for i in range(len(candidates))}

        while True:
            tiles = [
                _render_tile(idx + 1, cam, cap)
                for idx, (cam, cap) in enumerate(zip(candidates, captures, strict=False))
            ]
            grid = _assemble_grid(tiles, rows=rows, cols=cols)
            try:
                cv2.imshow(window_title, grid)
                key = cv2.waitKey(30) & 0xFF
            except cv2.error as exc:
                raise _GuiUnavailableError(str(exc).splitlines()[-1] if str(exc) else "no GUI") from exc
            if key in (ord("q"), 27):  # 'q' or ESC
                return None
            if key in digit_keys:
                return candidates[digit_keys[key]]
            # Detect window-close-button click.
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        for cap in captures:
            if cap is not None:
                cap.release()
        with contextlib.suppress(cv2.error):
            cv2.destroyWindow(window_title)


def _open_capture(cam: DiscoveredCamera):
    """Open a video capture for one camera, configured for low-bandwidth preview."""
    if cam.capture_node is None:
        return None
    import cv2

    cap = cv2.VideoCapture(cam.capture_node, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _TILE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _TILE_HEIGHT)
    return cap


def _grid_shape(n: int) -> tuple[int, int]:
    """Return (cols, rows) for an n-tile grid. Prefer roughly square layouts."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def _blank_tile(text: str):
    import cv2
    import numpy as np

    tile = np.zeros((_TILE_HEIGHT, _TILE_WIDTH, 3), dtype=np.uint8)
    tile[:] = (40, 40, 40)
    cv2.putText(tile, text, (12, _TILE_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return tile


def _render_tile(number: int, cam: DiscoveredCamera, cap):
    """Read one frame from ``cap`` and overlay number/caption onto it."""
    import cv2

    if cap is None:
        tile = _blank_tile("no capture node")
    else:
        ok, frame = cap.read()
        if not ok or frame is None:
            tile = _blank_tile("no frame")
        else:
            # Force exact tile dimensions in case the camera ignored our request.
            tile = cv2.resize(frame, (_TILE_WIDTH, _TILE_HEIGHT))

    # Big number, top-left, with dark drop-shadow so it reads on any background.
    digit = str(number)
    org = (16, 70)
    cv2.putText(tile, digit, org, cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(tile, digit, org, cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 255, 0), 3, cv2.LINE_AA)

    # Caption strip at the bottom: model + usb path.
    caption = f"{cam.model}  {cam.usb_path}"
    if len(caption) > 60:
        caption = caption[:57] + "…"
    cv2.rectangle(
        tile,
        (0, _TILE_HEIGHT - 22),
        (_TILE_WIDTH, _TILE_HEIGHT),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        tile,
        caption,
        (6, _TILE_HEIGHT - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def _assemble_grid(tiles, *, rows: int, cols: int):
    """Combine ``tiles`` into a single image of ``rows × cols`` cells.

    Missing cells (when ``len(tiles) < rows * cols``) get a blank filler tile.
    """
    import numpy as np

    filler = _blank_tile("")
    while len(tiles) < rows * cols:
        tiles = list(tiles) + [filler]
    row_arrays = []
    for r in range(rows):
        row_arrays.append(np.hstack(tiles[r * cols : (r + 1) * cols]))
    return np.vstack(row_arrays)
