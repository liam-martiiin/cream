# Unique Robot Identifier

## Problem

When we set up SO-101 arms today, robot identity is tied to two unstable things:

1. A name the user types on the CLI (`--robot.id=leader`).
2. The USB port the OS happens to assign (`/dev/ttyACM0`).

Swap the two USB cables and LeRobot will silently load the *leader's* calibration onto the *follower* — same `robot.id`, different physical hardware, no error. Result: wasted time, bad motion, occasional damage.

## Goal

Make the **physical board's hardware serial** the source of truth for robot identity. Friendly names, calibration files, and CLI flags all hang off that. Port paths become an implementation detail the user never has to type.

## Key finding

The Waveshare bus-servo boards used on SO-101 arms ship with a WCH CH343 USB-to-serial chip (USB ID `1a86:55d3`) that exposes a factory-burned serial in its USB descriptor. Confirmed on this machine with `udevadm`:

| Port | Serial |
|---|---|
| `/dev/ttyACM0` | `5AB9065381` |
| `/dev/ttyACM1` | `5B42134473` |

These serials persist across reboots, replugs, and different host machines. They're also exposed cross-platform via `serial.tools.list_ports.comports()` (pyserial), so no `udevadm` shell-outs are needed.

The Feetech STS3215 motors themselves have no hardware serial in the protocol — only a user-assignable ID byte — so the board is the right level to identify at. Since exactly one board sits in front of each arm's motor chain, identifying the board uniquely identifies the arm.

## Solution

1. **One-time per arm.** `lerobot-register-device --robot.type=so101_leader --name=leader` detects the board's serial, confirms with the user, and stores the mapping. If no calibration file exists for that name yet, it prompts to launch `lerobot-calibrate` immediately.

2. **From then on.** `lerobot-record --robot.id=leader` works without a `--robot.port` flag. The robot looks up the registered serial, scans connected USB serial devices, and resolves the right port automatically.

3. **On every connect.** The connected board's serial must match the serial stamped into the calibration file. Mismatch → hard error with a clear message explaining the likely cause (cables swapped, wrong arm physically connected, etc.) and pointing to the fix.

## Design decisions

- **Auto-resolve by serial.** The CLI no longer needs `--robot.port`; ports are discovered at connect time.
- **JSON registry next to calibrations.** Stored at `$HF_LEROBOT_CALIBRATION/devices.json`. Pure Python, no `sudo`, no udev rules, works the same on Linux / macOS / WSL.
- **Initial scope: SO-101 follower + leader only.** Same code generalizes to other Feetech-based robots, but we won't over-build until we need it.
- **Mismatch behavior: hard error.** Safer than warn-and-continue, friendlier than an interactive prompt that breaks scripts.
- **Port discovery via `pyserial`.** Already a transitive LeRobot dependency; cross-platform; exposes `.serial_number` directly.
- **Registry is the single source of truth — calibration files stay untouched.** The existing calibration JSON is loaded via `draccus.load(dict[str, MotorCalibration], f)`, which rejects any extra top-level keys. Rather than break that schema, the serial↔name binding lives only in `devices.json`. On connect, we look up `robot.id` in the registry and verify the connected board's serial matches. Calibration files are unchanged, no migration logic needed, existing users keep working until they choose to register.

## Features

- Unique board-serial-based ID, mapped to a friendly name and a calibration file.
- Optional immediate calibration launch after registration.
- Clear, actionable error messages for the common failure modes: cable swap, missing board, permission denied (e.g., user not in `dialout` group on Linux).
- Graceful upgrade for pre-existing calibration files.
- Built in the existing LeRobot style: draccus configs, a `src/lerobot/scripts/` entry point registered in `pyproject.toml`, dataclass utilities, type annotations consistent with neighboring modules.

## Out of scope (deferred, not forgotten)

- Multi-machine portability of the registry (sync via HF Hub).
- Cameras and Dynamixel boards — different chips, different discovery paths.
- udev rule generation for `/dev/lerobot/<name>` symlinks — the registry alone is sufficient.
- Generalization beyond SO-101 to all Feetech-based robots — trivial once the core lands.
