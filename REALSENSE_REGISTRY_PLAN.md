# Camera Registry: RealSense support + serial-first identification — Plan

## Goal

Let `lerobot-register-camera` discover **all** cameras (not just 2D V4L2/UVC), and identify
cameras with a **unique serial** (Intel RealSense, and any real camera) by that serial —
**port-independent**, so replugging doesn't force a re-pair. RealSense becomes referenceable
by a friendly registry `id` in configs, exactly like the OpenCV cameras.

## Key facts that shape this

- **RealSense has globally-unique serials.** `RealSenseCamera.find_cameras()` (pyrealsense2 SDK)
  reports each device's `serial_number`. So RealSense can be keyed by serial alone — no `usb_path`.
- **The UVC side already does serial-first.** `CameraRegistry.resolve()` Case 2 already silently
  self-heals a unique-serial UVC camera that moved ports. So the *new* work is RealSense, plus
  making the model/CLI/config aware of camera "kind".
- **`pyrealsense2` is an optional extra** (`lerobot[intelrealsense]`, guarded by `require_package`).
  All RealSense discovery/resolution must lazily import it and **degrade to "no RealSense found"**
  when it's absent — never crash. Tests mock it.
- **Hard limitation (out of scope, by physics):** the Innomaker webcams all report `serial=SN0001`.
  Nothing can make those port-independent — there's no unique ID. They keep the `usb_path` +
  picker behavior. Only unique-serial cameras (RealSense, etc.) benefit.

## Phase 0 — Principles

- Backward compatible: existing `cameras.json` entries (no `kind`) load as `kind="opencv"`.
- RealSense code paths are lazy + guarded; CI/tests without pyrealsense2 still pass (mocked).
- No change to UVC resolution behavior except adding the explicit `kind` dispatch.

## Phase 1 — Data model: add `kind`

**Files:** `src/lerobot/utils/camera_registry.py`

- `RegisteredCamera`: add `kind: str = "opencv"` (values `"opencv"` | `"intelrealsense"`).
  Default makes old JSON load unchanged.
- `DiscoveredCamera`: add `kind`. For RealSense, `serial` is the unique id, `usb_path` holds the
  SDK `physical_port` (a hint only), `capture_node=None`.

**Tests:** old-format JSON loads as `opencv`; round-trip preserves `kind`.

## Phase 2 — Discovery: unify UVC + RealSense

**Files:** `camera_registry.py`

- New `discover_realsense_cameras()`: lazily import pyrealsense2 (guarded); call
  `RealSenseCamera.find_cameras()`; map each to `DiscoveredCamera(kind="intelrealsense",
  serial=<serial>, model=<name>, usb_path=<physical_port>, capture_node=None)`. Returns `[]`
  if pyrealsense2 is missing or no device is present (no raise).
- New `discover_cameras()` = `discover_uvc_cameras()` + `discover_realsense_cameras()`, with
  **de-duplication**: a RealSense also enumerates as UVC nodes (Intel `vid=0x8086`, empty serial).
  Drop those UVC entries so the device appears once, as its `intelrealsense` entry.
- Keep `discover_uvc_cameras()` as-is for back-compat.

**Tests:** mocked RS enumeration → discovered with `kind` + serial; pyrealsense2 absent → empty,
no crash; Intel UVC nodes de-duped against the RS entry.

## Phase 3 — Resolution: dispatch by kind

**Files:** `camera_registry.py`

- `resolve(name, picker=None)` dispatches on `entry.kind`:
  - **`intelrealsense`** → `resolve_realsense(entry)`: confirm a connected RealSense has
    `entry.serial` (via `discover_realsense_cameras()`); return the **serial string**
    (port-independent; no usb_path, no picker). Raise `CameraNotConnectedError` if not present.
  - **`opencv`** → existing UVC logic unchanged (exact usb_path → unique-serial self-heal → picker).
- Document the return contract: `resolve()` yields a `/dev/video*` for `opencv` and a **serial**
  for `intelrealsense`; each camera backend consumes the right one.

**Tests:** RealSense entry resolves to its serial even after `physical_port` changes; not-connected
→ raises; the OpenCV/SN0001 paths are byte-for-byte unchanged (existing tests stay green).

## Phase 4 — Config: reference RealSense by registry `id`

**Files:** `src/lerobot/cameras/realsense/configuration_realsense.py`, `camera_realsense.py`

- `RealSenseCameraConfig`: add `id: str | None = None`; `__post_init__` enforces **exactly one**
  of `id` / `serial_number_or_name` (mirrors OpenCV's `id`/`index_or_path` rule).
- `RealSenseCamera.__init__`: if `config.id` is set, `serial = CameraRegistry.load().resolve(id)`
  (picker=None) and use it as `serial_number_or_name`. Lazy import of the registry.

**Tests:** config with `id` resolves to a serial; mutual-exclusion validation raises clearly.

## Phase 5 — CLI: register RealSense (and list everything)

**Files:** `src/lerobot/scripts/lerobot_register_camera.py`, `camera_registry.register(...)`

- Discovery switches to `discover_cameras()` (UVC + RealSense), each row labeled with its `kind`.
- RealSense has a unique serial → **no picker disambiguation**; register directly by serial.
  The preview grid stays UVC-only (RealSense shown as a labeled, pickable text entry).
- `register(...)` gains `kind` (carried from the discovered camera). The `replace=True` re-pair
  logic added earlier still applies, kind-aware.

**Tests:** registering a mocked RealSense stores `kind="intelrealsense"` + serial and resolves;
existing UVC register/re-pair tests stay green.

## Phase 6 — Verify

- Test suites: `tests/utils/test_camera_registry.py`, `tests/scripts/test_lerobot_register_camera.py`,
  `tests/cameras/test_realsense.py` (config `id`).
- `ruff check` + `ruff format` on all touched files.
- Manual smoke (you, with hardware): `lerobot-register-camera` lists the RealSense → register it →
  reference `{type: intelrealsense, id: <name>}` in a record/teleop command → replug the RealSense
  into a different USB port → it still resolves with **no re-pair**.

## Migration note

Your existing `top_depth_middle` entry was registered via the UVC path with `serial=''` (useless
for RealSense). After this lands, unregister it and re-register the RealSense through the new
RS-aware flow so it gets its real serial. One-time, manual.

## Out of scope / risks

- **SN0001 webcams stay port-bound** — physical limitation, no fix.
- **Generic multi-backend** (zmq, reachy2) — not included (you chose RealSense + serial-first).
- **New return semantics** for `resolve()` (serial vs node by kind) — internal, documented; the
  only behavior change, and it doesn't touch the OpenCV path.
- pyrealsense2 optional → every RealSense path guarded; verified absent-safe in tests.
