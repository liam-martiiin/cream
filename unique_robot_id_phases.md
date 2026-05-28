# Unique Robot ID — Implementation Phases

Each phase is small, independently testable, and leaves the codebase in a working state. Stop after any phase and previous behavior still works.

See [unique_robot_id.md](./unique_robot_id.md) for the design and decisions this plan implements.

> **Design refinement (2026-05-28):** Calibration JSON is loaded via `draccus.load(dict[str, MotorCalibration], f)` at [src/lerobot/robots/robot.py:160](src/lerobot/robots/robot.py#L160) — any extra top-level key would crash type validation. So the serial↔name binding lives **only** in `devices.json`. Calibration files stay unchanged, no migration needed.

---

## Phase 1 — Device registry utility

Pure-Python module. No CLI, no robot I/O. Owns the `devices.json` file and the pyserial scan.

**Add:** `src/lerobot/utils/device_registry.py`

- `RegisteredDevice` dataclass: `serial`, `name`, `robot_type`, `registered_at` (ISO timestamp).
- `DeviceRegistry`:
  - `load()` / `save()` — JSON read/write at `$HF_LEROBOT_CALIBRATION/devices.json`.
  - `register(serial, name, robot_type)` — add or update.
  - `find_by_name(name)`, `find_by_serial(serial)`.
  - `list_connected_boards()` — wraps `serial.tools.list_ports.comports()`, filters to VID:PID `1a86:55d3` (CH343), returns `(port, serial_number)` tuples.
  - `resolve_port(name)` — registry lookup + connected scan → `/dev/...` path. Raises `DeviceNotConnectedError` with a friendly message if the registered board isn't plugged in.
- Custom exceptions in the same module: `DeviceNotConnectedError`, `DeviceNameConflictError`, `DeviceMismatchError`.

**Add:** `tests/utils/test_device_registry.py`

- Round-trip add → save → load → find.
- Update existing entry (same serial, new name).
- Name conflict (same name, different serial) raises `DeviceNameConflictError`.
- `resolve_port` returns the right path when board is connected.
- `resolve_port` raises the right error when board is missing.
- Tests inject a fake `comports()` — no hardware required to run.

**Done when:** module imports cleanly, all unit tests pass. Nothing else in the codebase changes.

---

## Phase 2 — `lerobot-register-device` CLI

The user-facing entry point. Detects boards, registers them, optionally chains into calibration.

**Add:** `src/lerobot/scripts/lerobot_register_device.py`
**Modify:** `pyproject.toml` `[project.scripts]` — add `lerobot-register-device = "lerobot.scripts.lerobot_register_device:main"`.

Interactive flow:

```
$ lerobot-register-device --robot.type=so101_leader --name=leader
Scanning USB devices…
Found 2 SO-101 boards:
  1. /dev/ttyACM0  serial 5AB9065381  (unregistered)
  2. /dev/ttyACM1  serial 5B42134473  (already registered as "follower")

Register /dev/ttyACM0 (5AB9065381) as "leader" of type so101_leader? [Y/n]
✓ Registered.

No calibration file found for "leader". Run calibration now? [Y/n]
→ launches `lerobot-calibrate --teleop.type=so101_leader --teleop.id=leader`
```

Error paths handled explicitly with actionable messages:

- No SO-101 boards detected → "Plug in a board and re-run."
- All boards already registered → list current mappings, offer to re-pair one.
- Name already taken by a different serial → "Robot 'leader' currently maps to <other serial>. Re-pair? [Y/n]"
- Permission denied on serial port → on Linux, suggest `sudo usermod -aG dialout $USER && newgrp dialout`.

Uses draccus for argument parsing, matching the style of other scripts in `src/lerobot/scripts/`.

**Done when:** end-to-end registration works, `devices.json` is valid, calibration chaining launches the right `lerobot-calibrate` command.

---

## Phase 3 — Auto-resolve port + verify serial at connect time

Wire the registry into the SO-101 robot/teleop classes. This is the phase that gives the user the typing-reduction *and* the silent-mismatch detection — in one place.

**Modify:**

- `src/lerobot/robots/so_follower/so_follower.py` — in `connect()`, before opening the bus:

  ```python
  registry = DeviceRegistry.load()
  registered = registry.find_by_name(self.id)
  if registered is not None:
      # Registered path: auto-resolve port AND verify physical board.
      resolved_port = registry.resolve_port(self.id)
      connected_serial = serial_for_port(resolved_port)
      if connected_serial != registered.serial:
          raise DeviceMismatchError(self.id, registered.serial, connected_serial)
      self.bus.port = resolved_port
  # Unregistered fallback: use self.config.port as today. No behavior change.
  ```

- `src/lerobot/teleoperators/so_leader/so_leader.py` (or the equivalent SO-101 leader entry point) — same one-block addition.
- `src/lerobot/robots/so_follower/config_so_follower.py` and the SO-101 leader config equivalent — make `port: str | None = None` so the CLI flag becomes optional. Validation moves into `connect()`: if the robot isn't registered AND no port was passed, raise a clear "either pass --robot.port or register the device with `lerobot-register-device`" error.

**Done when:** this command works for registered devices with no port flags:

```
lerobot-record \
  --robot.type=so101_follower --robot.id=follower \
  --teleop.type=so101_leader --teleop.id=leader \
  --dataset.repo_id=<...>
```

And manually swapping the two USB cables triggers `DeviceMismatchError` instead of bad motion.

---

## Phase 4 — Docs, polish, end-to-end manual check

**Modify:**

- `AGENT_GUIDE.md` — add a "First-time SO-101 setup" section: register each arm once, then show the simplified `lerobot-calibrate` / `lerobot-record` / `lerobot-teleoperate` commands with no port flags.
- `docs/source/so101.mdx` (if present) — same simplification.

**End-to-end manual test** (requires hardware):

1. Register both arms with friendly names.
2. Power-cycle the host, swap the two USB cables.
3. Run `lerobot-record` — should auto-resolve each arm to the correct port. No port flags used.
4. Manually edit one device's `devices.json` entry to a bogus serial → confirm `DeviceMismatchError` fires with a clear message.
5. Unplug one arm and run a command that needs it → confirm `DeviceNotConnectedError` fires with a clear message.

**Done when:** docs match behavior and all manual checks pass.

---

## Risk notes

Worth flagging now so they don't bite during implementation:

- **Board swap, same motors.** If someone moves the controller board from arm A's motors to arm B's motors, the registry will say "this is leader" but the mechanical homing may differ slightly. We can't detect this from serial alone. Mention it once in the `lerobot-register-device` confirmation prompt and move on.
- **macOS port naming.** macOS uses `/dev/cu.usbmodem*` instead of `/dev/ttyACM*`. pyserial's `comports()` abstracts this — we should not hardcode `/dev/ttyACM` anywhere.
- **`HF_LEROBOT_CALIBRATION` may not exist yet.** Phase 1's `save()` needs to `mkdir -p` the parent on first write.
- **Existing users.** Anyone not registered keeps the current behavior (must pass `--robot.port`). They opt in by running `lerobot-register-device`.
