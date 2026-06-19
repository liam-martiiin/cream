# Batch interactive camera registration — Plan

## Goal

One command that shows **every** detected camera at once and lets you **assign a
name to each in a single session**, saving each by its best identifier:

- **RealSense** (and any camera with a unique serial) → keyed by **serial**, so it's
  port-independent (no re-pair on replug).
- **Look-alike webcams** that share a serial (Innomaker `SN0001`) → keyed by
  `usb_path`, with a **live preview** so you can tell them apart while naming.

This is additive UX on top of the existing registry — the per-camera
`lerobot-register-camera --name X` flow keeps working unchanged.

## Prerequisite (not code)

RealSense discovery needs the **`pyrealsense2`** SDK: `uv sync --extra intelrealsense`.
Without it, RealSense simply won't appear (graceful); everything else still works.

## Note on "other cameras with unique IDs"

Already handled: the resolver self-heals a unique-serial camera that moved ports, so
any camera with a genuinely unique serial is already port-independent. Only the
shared-serial Innomakers must stay port-bound (no unique ID to key on). The batch
flow inherits this automatically — no new resolution logic needed.

## Phase 1 — Batch assigner in `camera_picker.py`

**File:** `src/lerobot/utils/camera_picker.py`

Snapshot + Rerun display (works with headless OpenCV — no `cv2.imshow`, no files to open).

- `_snapshot_grid(cameras) -> np.ndarray | None`: grab ONE frame from each UVC camera
  (`cv2.VideoCapture(...).read()` — works headless), render numbered/captioned tiles
  (reuses `_render_tile`/`_assemble_grid`), assemble into one grid image. RealSense
  (no V4L2 node) gets a labeled "RealSense <model> serial=…" placeholder tile.
- `_show_grid(grid) -> bool`: lazily import `rerun`; `rr.init(spawn=True)` to auto-open
  the viewer and `rr.log(...)` the grid image. Returns False if `rerun` isn't installed
  (caller prints a fallback note). No manual file/folder opening.
- `assign_names(cameras) -> list[tuple[DiscoveredCamera, str]]`: build+show the snapshot
  grid, then per-camera terminal prompts — `Camera [i] <model> (usb_path / serial,
  kind): name (blank to skip):`. Blank skips. Display is best-effort/guarded so naming
  still works when Rerun or a snapshot is unavailable.
- The single-pick `pick_camera` path is unchanged.

**Tests:** mock `input` + stub the snapshot/Rerun display → correct (camera, name)
pairs; blank skips; no GUI/Rerun needed in tests.

## Phase 2 — CLI: `--all` batch mode

**File:** `src/lerobot/scripts/lerobot_register_camera.py`

- Make `--name` optional; add `--all`. Require **exactly one** of `--name` / `--all`.
- `--all` flow: `discover_cameras()` → `assign_names(...)` → for each (camera, name):
  validate, then `registry.register(camera, name, replace=True)` (re-pair aware,
  kind carried through). Then `save()` once and print a summary table
  (name → kind → serial/usb_path).
- Validations: reject **duplicate names within one batch** (clear error, nothing
  saved); `--force` skips the per-camera re-pair confirmation; `--headless` forces text.
- Reuse existing permission-error and no-cameras handling.

**Tests:** `--all` registers multiple cameras including a (mocked) RealSense by serial
and webcams by usb_path; blank-name cameras are skipped; duplicate names rejected;
`--name`/`--all` mutual-exclusion enforced.

## Phase 3 — Verify

- Suites: `tests/scripts/test_lerobot_register_camera.py`, `tests/utils/test_camera_picker.py`,
  `tests/utils/test_camera_registry.py`. `ruff check` + `ruff format`.
- Manual (you, with `pyrealsense2` installed + cameras connected):
  `lerobot-register-camera --all` → all cameras shown (incl RealSense) → assign names →
  replug the RealSense into a different port → it still resolves with no re-pair.

## Out of scope / decisions

- **No live preview for RealSense** in the grid (it has no V4L2 node; previewing it
  would need an `rs.pipeline`). It's shown as a labeled tile/text entry — pickable by
  index. Could be added later.
- Existing single `--name` flow and all current resolution behavior are unchanged.
- Duplicate names within a batch are rejected rather than silently last-wins.
