# Unique Camera ID — Implementation Phases

Each phase is small, independently testable, and leaves the codebase in a working state. Stop after any phase and previous behavior still works.

See [unique_camera_id.md](./unique_camera_id.md) for the design and decisions this plan implements.

> **Status (2026-05-28):** SO-101 board registry is shipped (see [unique_robot_id.md](./unique_robot_id.md)). This plan reuses the same patterns where possible — registry under `$HF_LEROBOT_CALIBRATION`, optional `id` field on the device config, auto-resolve at construction time.

---

## Phase 1 — Camera registry utility

Pure-Python module. No CLI, no GUI, no V4L2 capture. Just discovery, registry CRUD, and the compound-key match logic.

**Add:** `src/lerobot/utils/camera_registry.py`

- `RegisteredCamera` dataclass: `name`, `vid`, `pid`, `serial`, `usb_path`, `model`, `registered_at`.
- `DiscoveredCamera` dataclass: a UVC device currently plugged in, with its physical-camera identifiers and the list of `/dev/video*` nodes it owns (one camera ⇒ multiple V4L2 nodes).
- `discover_uvc_cameras()` — walk `udevadm info --query=property` (or pyudev if a dep is acceptable) for every `/dev/video*` node, group nodes by `ID_PATH` so multi-node cameras collapse into one logical entry. Returns `list[DiscoveredCamera]`.
- `find_capture_node(camera: DiscoveredCamera) -> str | None` — among a camera's V4L2 nodes, pick the one with `V4L2_CAP_VIDEO_CAPTURE`. Implemented via the `VIDIOC_QUERYCAP` ioctl (use `fcntl` + `ctypes`, no extra dep) or by parsing `v4l2-ctl --device=… --info` output as a portable fallback.
- `CameraRegistry`:
  - `load()` / `save()` — JSON at `$HF_LEROBOT_CALIBRATION/cameras.json`.
  - `register(camera: DiscoveredCamera, name: str)`, `find_by_name`, `unregister`.
  - `resolve(name: str, picker: Callable | None = None) -> str` — the heart of the feature:
    1. Look up the registry entry for `name`.
    2. Discover all connected UVC cameras.
    3. If exactly one match on `(vid, pid, serial, usb_path)` → return its capture node.
    4. If `(vid, pid, serial)` matches multiple and the registered `usb_path` is gone → call `picker(candidates)` to re-bind, persist the new `usb_path`, return the new capture node.
    5. If nothing matches → raise `CameraNotConnectedError`.
- Custom exceptions: `CameraNotConnectedError`, `CameraNameConflictError`, `NoCaptureNodeError`.

**Add:** `tests/utils/test_camera_registry.py`

- Discovery groups multi-node cameras correctly (mock `udevadm` output with two cameras × two nodes each).
- Registry CRUD round-trip.
- `resolve()` happy path (exact usb_path match) returns the capture node.
- `resolve()` picker-fallback path (multiple matches on `(vid,pid,serial)`, usb_path gone) calls the picker and updates the registry.
- `resolve()` hard-error path (nothing matches) raises with a clear message.
- Capture-node detection: when two nodes share a camera, the one without `V4L2_CAP_VIDEO_CAPTURE` is skipped.
- Tests inject a fake discovery function and a fake picker so no hardware or display is required.

**Done when:** module imports cleanly, all unit tests pass. Nothing else in the codebase changes.

---

## Phase 2 — Grid picker UI

Standalone interactive picker. Reusable from both the registration CLI and the connect-time self-healing path.

**Add:** `src/lerobot/utils/camera_picker.py`

- `pick_camera(candidates: list[DiscoveredCamera], prompt: str) -> DiscoveredCamera | None`.
- Implementation:
  - Open one `cv2.VideoCapture` per candidate. Lower the requested resolution to ~320×240 to stay within USB 2.0 bandwidth when N is large.
  - In a loop: grab one frame from each, tile into a single `numpy.ndarray` grid (auto-sizing to the candidate count), overlay each tile with a big `1`–`N` number plus its model name and usb-path.
  - Display via `cv2.imshow(prompt, grid)`, listen with `cv2.waitKey(30)`.
  - Keys: `1`–`N` (or `q` to cancel). Return the chosen candidate, or `None` on cancel.
  - Clean up: release all captures and destroy the window in a `finally` block.
- **Headless fallback:** if `LEROBOT_HEADLESS_PICKER=1` is set, or `cv2.imshow` raises (no display server), print a numbered text list to stdout and read the choice from stdin.
- **No new optional dep:** opencv-python-headless is already in the base requirements, so this works without `[hardware]`/`[opencv]` extras.

**Add:** `tests/utils/test_camera_picker.py`

- Headless mode: feed stdin, assert the right candidate is returned.
- Cancellation: stdin = `q` → returns `None`.
- Invalid input: stdin = `9` (out of range) → re-prompts until valid.
- GUI-mode tests skip on CI (no display); add `@pytest.mark.skipif(os.environ.get('DISPLAY') is None)` guard.

**Done when:** picker is callable from a Python shell against real hardware and the headless fallback path is unit-tested.

---

## Phase 3 — `lerobot-register-camera` CLI

The user-facing entry point. Discovers, displays, registers.

**Add:** `src/lerobot/scripts/lerobot_register_camera.py`
**Modify:** `pyproject.toml` `[project.scripts]` — add `lerobot-register-camera = "lerobot.scripts.lerobot_register_camera:main"`.

Interactive flow (single-name registration):

```
$ lerobot-register-camera --name right_overhead
Scanning UVC cameras…
Found 3 cameras:
  [1] /dev/video0  Innomaker-U20CAM-1080p-S1  usb-0:4:1.0  (unregistered)
  [2] /dev/video2  Innomaker-U20CAM-1080p-S1  usb-0:3:1.0  (unregistered)
  [3] /dev/video4  Innomaker-U20CAM-1080p-S1  usb-0:2:1.0  (registered as "wrist_cam")

Opening grid picker… press 1–3 to bind "right_overhead", or q to cancel.

✓ Registered "right_overhead" → Innomaker-U20CAM at usb-0:4:1.0 (/dev/video0).
  Saved to /home/peter/.cache/huggingface/lerobot/calibration/cameras.json
```

Already-registered cameras appear in the grid dimmed and labeled with their current name; picking one of them triggers a "re-pair?" confirmation. Mirrors the SO-101 register-device flow.

Args:
- `--name <friendly_name>` (required) — what to call the chosen camera.
- `--force` — skip the re-pair confirmation when the chosen camera is already registered.
- `--headless` — force the text-mode fallback even if a display is available.

Error paths handled with actionable messages:
- No UVC cameras detected → "Plug a camera in and re-run."
- Permission denied opening `/dev/video*` → on Linux, suggest `sudo usermod -aG video $USER && newgrp video`.
- Picked camera has no capture-capable V4L2 node → "This device exposes only metadata streams; not a usable camera."

**Add:** `tests/scripts/test_lerobot_register_camera.py`

- Argparse rejects missing `--name`.
- Successful registration round-trips through the registry.
- Cancellation (`q` in picker) returns non-zero exit code, registry unchanged.
- Re-pair confirmation flow.
- Tests inject a fake picker so no GUI or hardware is needed.

**Done when:** end-to-end registration works against real hardware and `cameras.json` is valid.

---

## Phase 4 — Wire auto-resolve into `OpenCVCameraConfig`

Make the existing camera config recognize `id` and auto-resolve via the registry.

**Modify:** `src/lerobot/cameras/opencv/configuration_opencv.py` (or whatever the exact config file is — verify path at impl time).

- Add `id: str | None = None` field.
- Add post-init validation: `index_or_path` and `id` are mutually exclusive; exactly one must be set.

**Modify:** `src/lerobot/cameras/opencv/camera_opencv.py` (the runtime class).

- In `__init__`: if `config.id` is set, call `CameraRegistry.load().resolve(config.id, picker=pick_camera)` and assign the returned `/dev/video*` path to whatever internal field the OpenCV camera uses to open its capture.
- The `picker=pick_camera` argument is what enables self-healing: if the usb-path no longer matches, the picker fires inline and the user re-binds without exiting the program.

**Modify:** `OpenCVCameraConfig` YAML/JSON usage — no breaking changes; existing users keep using `index_or_path`. New users use `id`.

**Add tests** in the existing opencv camera test file:

- Config validation: setting both `id` and `index_or_path` raises.
- Config validation: setting neither raises.
- Camera init with `id`: registry resolution is invoked, the resolved path is used to open the capture (mock the actual `cv2.VideoCapture` so no hardware required).

**Done when:** this YAML works against registered cameras with no `index_or_path`:

```python
--robot.cameras="{ front: {type: opencv, id: right_overhead, width: 1920, height: 1080, fps: 30}}"
```

And physically swapping which USB port `right_overhead` is plugged into triggers the picker on the next connect, transparently updates the registry, and the run continues.

---

## Phase 5 — Docs, polish, end-to-end manual check

**Modify:**

- `AGENT_GUIDE.md` §4.2 — extend the "Register the arms" section to also cover `lerobot-register-camera`. Update the `--robot.cameras=` examples in §4.5, §4.6, §4.10 to show the `id`-based form alongside the legacy `index_or_path` form.
- `docs/source/so101.mdx` and any other places that document camera config — same simplification.

**End-to-end manual test** (requires hardware):

1. Register two cameras with names like `right_overhead` and `left_overhead`.
2. Confirm `lerobot-teleoperate --robot.cameras="{ front: {type: opencv, id: right_overhead, ...} }"` opens the right physical camera with no `index_or_path`.
3. Power-cycle the host. Re-run — should resolve correctly without any prompt.
4. **Swap the two cameras' USB cables between ports.** Re-run — the picker should fire automatically, you press `1` or `2`, and the run continues. Verify the registry's `usb_path` field has been updated for both cameras.
5. Unplug one camera entirely. Confirm `CameraNotConnectedError` fires with a clear message ("`right_overhead` is registered but no matching camera is connected").
6. Set `LEROBOT_HEADLESS_PICKER=1` and repeat the cable-swap test over SSH. Confirm the text-mode fallback works.

**Done when:** docs match behavior and all manual checks pass.

---

## Risk notes

Worth flagging now so they don't bite during implementation:

- **USB 2.0 bandwidth ceiling during the picker.** Six 1080p MJPEG streams won't fit on a single USB 2.0 controller. The picker must request a low resolution (e.g. 320×240) on each `VideoCapture`. The actual recording configuration is independent and stays at full resolution.
- **Multi-node UVC cameras.** Each Innomaker U20CAM exposes two `/dev/video*` nodes. The discovery layer must group them by `ID_PATH` and pick the one with `V4L2_CAP_VIDEO_CAPTURE`. Skipping this means we sometimes try to open metadata streams as cameras, which fails opaquely.
- **`cv2.imshow` needs a display server.** Headless invocations (over SSH, in a Docker container without X) will hit this. The text-fallback path must be tested, and `LEROBOT_HEADLESS_PICKER=1` should be honored before any `imshow` call so we don't crash mid-pick.
- **Permission errors on Linux.** `/dev/video*` requires the user be in the `video` group (similar to `dialout` for serial). Catch `PermissionError` and suggest the fix, mirroring how the SO-101 register-device script handles `dialout`.
- **Discovery on macOS.** `udevadm` doesn't exist; we'd need to use `system_profiler SPUSBDataType` or `ioreg` to get USB topology. **Initial scope is Linux only** for camera registration — macOS users keep using `index_or_path` until a follow-up phase. Document this clearly.
- **`HF_LEROBOT_CALIBRATION` may not exist yet.** Phase 1's `save()` needs to `mkdir -p` the parent — same as the SO-101 registry.
- **Existing users.** Anyone not using `id` keeps the current behavior with `index_or_path`. They opt in by running `lerobot-register-camera`. No migration step required.
