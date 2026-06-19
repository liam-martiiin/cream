# Tier 1 Source-Code Fixes — Implementation Plan

Four robustness fixes distilled from a long bring-up debugging session on a
bimanual SO-101 setup. Each one removes a wall we actually hit. Scope is kept
deliberately tight; behavior changes and exclusions are called out explicitly.

| # | Fix | Symptom it removes |
|---|-----|--------------------|
| 1 | Bimanual configs get default arm-configs | `Missing required field(s) left_arm_config, right_arm_config`; forced `use_degrees=true` dummies |
| 2 | Best-effort torque-disable on disconnect | Ctrl+C → `RuntimeError: ... Overload error!` traceback; one arm's failure stranding the other |
| 3 | Camera resolution validation + Linux backend default | `failed to set capture_width=640 (actual_width=640, width_success=False)` |
| 4 | Camera operation never pops an interactive picker | `teleoperate` hanging on an OpenCV grid waiting for a click |

Guiding principles: smallest change that fixes the root cause; preserve existing
behavior for callers who pass explicit values; add a focused regression test per
fix; no scope creep into the larger Tier-2 items.

---

## Phase 0 — Safety & baseline

- Confirm the working tree state and that the existing device-registry tests pass before touching anything.
- Make changes on the current branch as a reviewable diff (no commits unless asked).
- Each phase is independent and individually testable, so they can land/revert separately.

---

## Phase 1 — Fix #1: default arm-configs for bimanual

**Files**
- `src/lerobot/teleoperators/bi_so_leader/config_bi_so_leader.py`
- `src/lerobot/robots/bi_so_follower/config_bi_so_follower.py`

**Change**
- Give `left_arm_config` / `right_arm_config` a `field(default_factory=...)` (`SOLeaderConfig` / `SOFollowerConfig`). Both sub-configs are all-defaults, and the bases are `@dataclass(kw_only=True)`, so this is safe and order-independent.

**Effect**
- `--teleop.id=... --robot.id=...` works with no `use_degrees` dummies; explicit arm-configs still override.

**Exclusions (important)**
- `bi_rebot_102_leader`, `bi_rebot_b601_follower`, `bi_openarm_*` are **out of scope**: their sub-configs have required fields (e.g. `RebotArm102LeaderConfig.port: str`), so a bare `default_factory` would raise. Those need their own decision and are deferred.

**Tests**
- Extend `tests/teleoperators/test_bi_so_leader.py` and `tests/robots/test_bi_so_follower.py`: assert the config parses from `--id=...` alone (no arm-config args) and that both ports remain `None` (registry-resolved).

---

## Phase 2 — Fix #2: don't let `disable_torque` crash disconnect

**Files**
- `src/lerobot/motors/motors_bus.py` (`MotorsBus.disconnect`)
- `src/lerobot/robots/bi_so_follower/bi_so_follower.py` (`disconnect`)
- `src/lerobot/teleoperators/bi_so_leader/bi_so_leader.py` (`disconnect`)

**Change**
- In `MotorsBus.disconnect`, wrap `self.disable_torque(num_retry=5)` in try/except: on failure log a warning and **still** `closePort()`. A motor reporting Overload at shutdown must not abort cleanup.
- In the two bimanual `disconnect()` methods, disconnect both sub-arms even if the first raises (try/finally or guarded calls), so one arm's failure can't strand the other torque-on.

**Effect**
- Ctrl+C during a strained pose closes cleanly with a warning instead of a traceback; both arms always get torn down.

**Tests**
- `tests/motors/`: a `FeetechMotorsBus` with a mocked `port_handler`, monkeypatch `disable_torque` to raise → assert `disconnect()` does not raise and `closePort()` was still called.
- Bimanual: mock one sub-arm's `disconnect` to raise → assert the other sub-arm's `disconnect` is still called.

---

## Phase 3 — Fix #3: trust the readback + Linux backend default

**Files**
- `src/lerobot/cameras/opencv/camera_opencv.py` (`_validate_width_and_height`, `_validate_fps`)
- `src/lerobot/cameras/opencv/configuration_opencv.py` (`backend` default)

**Change**
- In `_validate_width_and_height` / `_validate_fps`, drop the `not success` gate and rely on the **readback** (`get(...)`): raise only when the actual value ≠ requested. Many V4L2/GStreamer drivers return `False` from `set()` yet apply the value (exactly our case). Log a debug note when `set()` returned `False` but the value matched.
- Change the `backend` default from `Cv2Backends.ANY` to **`V4L2` on Linux** (platform-aware `field(default_factory=...)`), since `ANY` selected a backend whose `set()` calls misbehaved.

**Behavior change (called out)**
- ~~New Linux default backend is V4L2.~~ **REVERTED during implementation.** Defaulting to V4L2 broke opening video *files* (V4L2 is device-only), failing pre-existing `test_opencv` tests. The validation relaxation alone already fixes the reported error (the `width_success=False` no longer raises under `ANY`), so the backend default is unnecessary. `backend` stays `ANY`; the docstring now recommends passing `backend=V4L2` explicitly for `/dev/video*` on Linux.

**Tests**
- `tests/cameras/`: construct an `OpenCVCamera` (index path, no registry), set a mocked `videocapture` whose `set()` returns `False` but `get()` returns the requested W/H/FPS → assert `_validate_width_and_height` / `_validate_fps` do **not** raise.
- Assert the config's default `backend` is `V4L2` on Linux.

---

## Phase 4 — Fix #4: camera operation resolves deterministically (no GUI prompt)

**Files**
- `src/lerobot/cameras/opencv/camera_opencv.py` (registry resolve call)
- `src/lerobot/utils/camera_registry.py` (ambiguity error message)

**Change**
- In `OpenCVCamera.__init__`, resolve the registry id with `picker=None` instead of `pick_camera`. During robot operation, an unresolvable name should raise a clear error, never pop an interactive grid. Interactive picking stays exclusive to `lerobot-register-camera`.
- Improve the `len(candidates) > 1` error in `CameraRegistry.resolve` to be user-actionable: name the camera, list the candidate `usb_path`s, and suggest `lerobot-register-camera --name <name>` / unplugging duplicates — instead of the current developer-facing "no picker was provided / re-run with a picker callback."

**Rationale**
- The hang happened *with* a display present; gating on `DISPLAY` wouldn't have helped. The fix is to not invoke a picker during operation at all.

**Tests**
- `tests/utils/test_camera_registry.py` (or a sibling): register two same-serial cameras at different `usb_path`s with both connected → `resolve(name, picker=None)` raises with the candidate paths and the registration hint in the message.

---

## Phase 5 — Verify

- Run the touched test suites: `tests/teleoperators`, `tests/robots`, `tests/motors`, `tests/cameras`, `tests/utils`, `tests/scripts`.
- `ruff check` + `ruff format` on every changed file.
- Sanity-parse the real bimanual teleop command (no `use_degrees` dummies now) to confirm #1 end-to-end.

---

## Risks & rollback

- **#3 backend default** is the only behavior change with broad reach; mitigated by keeping `ANY` off-Linux and honoring explicit `backend=`. Easily reverted in isolation.
- **#4** changes an ambiguous-camera prompt into an error; this is the intended behavior for non-interactive commands and does not affect `lerobot-register-camera`.
- Each phase is a small, isolated diff with its own test, so any one can be reverted without disturbing the others.

## Out of scope (deferred Tier 2/3)

- Generalizing the device registry beyond SO-101 VID/PID (reBot support).
- RealSense registry-`id` resolution.
- Bimanual explicit per-arm registered names.
- Richer "no motors found" diagnostics.
