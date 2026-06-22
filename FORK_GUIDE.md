# SO-101 Lab Fork — what this adds on top of LeRobot

This is a fork of [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
customized for our **bimanual SO-101** bring-up. Everything upstream still works
exactly as documented in [`README.md`](./README.md) and
[`AGENT_GUIDE.md`](./AGENT_GUIDE.md); this file documents **only what we added or
changed**, so you don't have to diff against upstream to know what's ours.

The whole point of these additions: **refer to arms and cameras by friendly
names instead of fragile `/dev/ttyACM*` ports and `/dev/video*` paths that
renumber every time you replug or reboot.**

---

## TL;DR — the new commands

| Command | What it does | New? |
|---|---|---|
| `lerobot-register-device` | Pair an arm's controller board to a friendly name (by serial) | **New CLI** |
| `lerobot-register-device --unregister --name X` | Drop a device binding | **New flag** |
| `lerobot-register-camera --name X` | Pair one camera to a friendly name | **New CLI** |
| `lerobot-register-camera --all` | Show every camera at once and name them all in one session | **New flag** |

After registering, you drive everything with `--robot.id=` / `--teleop.id=` and
`{type: ..., id: ...}` camera specs — **no ports, no `/dev/video*` paths.**

State is stored as JSON under `$HF_LEROBOT_CALIBRATION` (default
`~/.cache/huggingface/lerobot/calibration/`):
- `devices.json` — name → board serial
- `cameras.json` — name → camera identity

These files are per-machine. Each lab computer registers its own hardware once.

---

## 1. Device registry — name your arms

Controller boards have **globally-unique USB serial numbers**, but their
`/dev/ttyACM*` ports are not stable (they depend on plug order / boot). The
device registry pins a friendly name to the **serial**, so the port is resolved
automatically no matter where you plug it.

### Register

```bash
# Single SO-101 arms
lerobot-register-device --type so101_leader   --name leader
lerobot-register-device --type so101_follower --name follower

# Skip the post-registration calibration offer
lerobot-register-device --type so101_follower --name follower --no-calibrate

# Remove a binding (e.g. after swapping hardware)
lerobot-register-device --unregister --name leader
```

Re-pairing a name to a *new* board is just running the register command again
and confirming the prompt — you don't have to unregister first.

### Use it

```bash
# Instead of --robot.port=/dev/ttyACM0, use the name:
lerobot-teleoperate \
  --robot.type=so101_follower --robot.id=follower \
  --teleop.type=so101_leader  --teleop.id=leader
```

### Safety nets (behavior change vs base LeRobot)

- **Serial verification:** if you pass `--robot.id=follower` and the board
  actually plugged in has a different serial, it raises a clear
  `DeviceMismatchError` instead of silently driving the wrong arm.
- **Cable-swap cross-check:** even if you pass a raw `--robot.port` (or an
  unregistered id), if the board at that port is registered under *another*
  name, it raises `BoardClaimedByAnotherNameError` ("Did you swap USB cables, or
  do you mean `--robot.id=<that name>`?"). The guarantee is symmetric — you can't
  accidentally drive the wrong arm whether you used an id or a port.

---

## 2. Camera registry — name your cameras (incl. RealSense)

Two kinds of cameras are handled differently, on purpose:

- **Cameras with a unique serial (Intel RealSense, and most decent cameras)** are
  keyed by **serial** → fully **port-independent**. Replug into any USB port and
  it still resolves — no re-pairing.
- **Cheap webcams that share a serial** (our Innomaker U20CAMs all report
  `iSerial=SN0001`) can't be told apart by serial, so they're keyed by
  `(vid, pid, serial, usb_path)` — i.e. *which physical USB port*. These need an
  interactive pick to disambiguate, and re-pairing if you move the cable.

### Register one camera

```bash
lerobot-register-camera --name right_overhead
lerobot-register-camera --name top_depth          # an Intel RealSense, if connected
lerobot-register-camera --name wrist_cam --headless   # force text mode (no preview)
lerobot-register-camera --name workspace --force      # skip the re-pair confirmation
```

### Register every camera at once (batch)

```bash
lerobot-register-camera --all
```

`--all` opens a **single snapshot of every detected camera in the Rerun viewer**
(auto-spawns; works with headless OpenCV — nothing to open manually), then
prompts you to name each one in a single session. Leave a name blank to skip a
camera. Duplicate names within one batch are rejected (nothing is saved).

> RealSense needs the SDK: `uv sync --extra intelrealsense`. Without it,
> RealSense simply doesn't appear — everything else still works.

### Use it

```bash
# UVC webcam:
--robot.cameras="{ front: {type: opencv, id: right_overhead, width: 640, height: 480, fps: 30} }"

# Intel RealSense (by registry id, port-independent):
--robot.cameras="{ top: {type: intelrealsense, id: top_depth, width: 640, height: 480, fps: 30} }"
```

### Behavior changes vs base LeRobot

- **No interactive picker during operation.** `teleoperate`/`record` will *never*
  pop a camera-picker grid and hang waiting for a click. If a name can't be
  resolved, it raises an actionable error telling you to replug or re-pair.
  Interactive picking is exclusive to `lerobot-register-camera`.
- **Resolution read-back is trusted.** Some V4L2/GStreamer drivers return
  `False` from `set(width/height/fps)` but actually apply the value. We now fail
  only on a real mismatch between requested and actual, fixing spurious
  `failed to set capture_width=... (width_success=False)` crashes.

---

## 3. Bimanual SO-101 quality-of-life

You can now drive a bimanual rig with **one id** and no per-arm port flags.

### Register both boards per side

```bash
lerobot-register-device --type so101_follower --name bimanual_left
lerobot-register-device --type so101_follower --name bimanual_right
lerobot-register-device --type so101_leader   --name bilead_left
lerobot-register-device --type so101_leader   --name bilead_right
```

The bimanual classes give each sub-arm an id of `{id}_left` / `{id}_right`
automatically, so register the boards under those exact suffixed names.

### Use it

```bash
lerobot-teleoperate \
  --robot.type=bi_so_follower --robot.id=bimanual \
  --teleop.type=bi_so_leader  --teleop.id=bilead
```

- **Default arm-configs:** `bi_so_follower` / `bi_so_leader` configs now default
  their `left_arm_config` / `right_arm_config`, so `--robot.id=` alone is enough
  — you no longer have to pass dummy `use_degrees` / port fields just to satisfy
  required sub-config fields. Explicit
  `--robot.left_arm_config.port=...` still overrides.
- **Robust disconnect:** both arms are always torn down even if one raises, and a
  motor reporting an Overload fault at shutdown no longer aborts cleanup with a
  traceback (it logs a warning and still closes the port). Ctrl-C during a
  strained pose exits cleanly.

> Note: only the **SO-101** bimanual variants get default arm-configs. Other
> bimanual types (`bi_rebot_*`, `bi_openarm_*`) have required sub-config fields
> and are intentionally left unchanged.

---

## 4. Where the state lives / sharing across machines

- `devices.json` and `cameras.json` live under `$HF_LEROBOT_CALIBRATION`
  (default `~/.cache/huggingface/lerobot/calibration/`).
- They are **machine-specific** (serials and USB topology differ per computer),
  so each lab machine should run the `register-*` commands once for its own
  hardware. Don't commit these files to the repo.

---

## 5. Keeping in sync with upstream LeRobot

This fork tracks `huggingface/lerobot`. To pull upstream updates:

```bash
git remote -v                       # 'upstream' should be huggingface/lerobot
git fetch upstream
git merge upstream/main             # resolve conflicts (most likely in src/lerobot/utils/*registry*)
```

Our additions are intentionally isolated (new files for the registries + small,
well-scoped edits to camera/bimanual code), so merge conflicts should be rare and
localized.

---

## 6. Implementation notes / provenance

All of the above is captured in three planning docs in the repo root, which
record the exact symptom each change removes:

- [`REALSENSE_REGISTRY_PLAN.md`](./REALSENSE_REGISTRY_PLAN.md) — RealSense + serial-first cameras
- [`CAMERA_BATCH_REGISTER_PLAN.md`](./CAMERA_BATCH_REGISTER_PLAN.md) — the `--all` batch flow
- [`TIER1_FIXES_PLAN.md`](./TIER1_FIXES_PLAN.md) — the four robustness fixes

Source of truth:
- `src/lerobot/utils/device_registry.py`, `src/lerobot/scripts/lerobot_register_device.py`
- `src/lerobot/utils/camera_registry.py`, `src/lerobot/utils/camera_picker.py`, `src/lerobot/scripts/lerobot_register_camera.py`
- `src/lerobot/robots/bi_so_follower/`, `src/lerobot/teleoperators/bi_so_leader/`
- `src/lerobot/cameras/opencv/`, `src/lerobot/cameras/realsense/`, `src/lerobot/motors/motors_bus.py`

Each change has regression tests under `tests/` (all passing).
