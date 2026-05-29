# Unique Camera Identifier

## Problem

LeRobot today identifies cameras by OpenCV index (`index_or_path: 0`) or raw device path (`/dev/video0`). Both are unstable:

- The kernel assigns `/dev/videoN` numbers in plug-detection order. Boot a different USB hub or unplug-replug, and the same physical camera ends up at a different number.
- OpenCV indices follow the same logic, so they reshuffle for the same reasons.
- A single physical camera typically creates **multiple** `/dev/video*` nodes (one for capture, one for metadata). Picking the right one isn't obvious from outside.

The practical result: every time the user plugs cameras into a different port, or reboots into a different state, they have to manually re-discover which `index_or_path` corresponds to which physical view, and edit their config accordingly.

## Goal

Make each physical camera addressable by a **friendly name** (`right_overhead`, `wrist_cam`) that survives reboots, replugs into different USB ports, and the V4L2-node reshuffling that comes with both. Mirror the workflow we already built for SO-101 boards — register once, then drop `index_or_path` from all subsequent commands.

## Key finding

The SO-101 case was easy because every controller board ships with a factory-burned unique USB serial number. **The Innomaker U20CAM-1080p does not have this property.**

Empirical check with two U20CAMs plugged in simultaneously:

| Field | Camera A | Camera B |
|---|---|---|
| `ID_VENDOR_ID` | `0c45` | `0c45` |
| `ID_MODEL_ID` | `6366` | `6366` |
| `ID_SERIAL_SHORT` | `SN0001` | `SN0001` ← **identical** |
| `ID_PATH` | `pci-…-usb-0:4:1.0` | `pci-…-usb-0:3:1.0` |

Every U20CAM ships with the hardcoded serial `SN0001`. This is common for cheap UVC webcams — the Sonix/Microdia silicon stores `iSerial` in writable EEPROM, but manufacturers rarely bother burning unique values per unit.

The *only* identifier that distinguishes the two cameras is **`ID_PATH`** — the physical USB topology path (which hub, which port). That works across reboots, but **breaks the moment a cable is moved to a different port**. So we can't treat it as a true identity; we have to treat it as a fast-path hint that the user might invalidate at any time.

## Solution

A two-layer identification strategy:

1. **Fast path: compound key lookup.** Each registered camera stores `(vid, pid, serial, usb_path)`. At connect time, scan all video nodes and find the one matching all four fields. For cameras with genuinely unique serials (RealSense, ELP industrial, etc.), this is the whole story.

2. **Self-healing path: interactive grid picker on mismatch.** If the `(vid, pid, serial)` triple matches multiple cameras (the U20CAM case) and the registered `usb_path` is no longer present, automatically launch a grid picker so the user can re-bind the friendly name to whichever physical camera it now points to. Update the registry transparently. No cryptic errors, no requirement to remember which usb-path was "right_overhead."

The grid picker — chosen because covering one of six lenses doesn't scale — opens a 3×N tiled live preview of all candidates with big number overlays, and the user presses **1**–**N** to bind the current friendly name. One keypress per camera, all cameras visible at once for verification.

## Design decisions

- **Compound key, not pure serial.** Necessary because cheap UVC silicon shares serials. RealSense and other industrial cameras with real serials degrade gracefully — they just never hit the picker fallback.
- **Self-healing on cable swap.** Hard-erroring on usb-path mismatch (as the SO-101 design does for serial mismatch) was rejected because the U20CAMs have no way to verify identity *other than* re-asking the user. So we re-ask, on the spot, with a 1-keypress UI.
- **Grid picker, not arrow-key carousel.** All cameras visible simultaneously; scales cleanly from 1 to 12+; 1 keypress per binding. The carousel was the user's original suggestion but doesn't scale past ~3 cameras.
- **JSON registry next to `devices.json`.** Stored at `$HF_LEROBOT_CALIBRATION/cameras.json`. Same rationale as the SO-101 registry: no sudo, no udev rules, cross-platform.
- **`id` field on `OpenCVCameraConfig`.** Mutually exclusive with `index_or_path`. Setting one triggers registry resolution; the other preserves legacy behavior. No migration required.
- **Initial scope: UVC cameras via OpenCV.** Covers Innomaker U20CAMs, most webcams. RealSense devices use the librealsense SDK and a different config class — out of scope for v1 but the same `id` pattern is trivially portable later.
- **Headless fallback.** When running over SSH or without a display server, the picker switches to a text listing (`[1] /dev/video0 — usb-0:4:1.0`) and reads stdin. Loses the visual verification, gains the ability to run remotely.

## Features

- Friendly camera names that persist across reboots, replugs, and cable swaps.
- One-time registration via an interactive grid picker — see all cameras at once, press a number to bind.
- Self-healing re-bind when a registered camera is no longer at its expected USB port.
- Clear, actionable error messages for the common failure modes: camera unplugged, ambiguous match in headless mode, V4L2 device not capture-capable.
- Built in the existing LeRobot style: a `src/lerobot/scripts/` entry point registered in `pyproject.toml`, optional `id` field on the existing `OpenCVCameraConfig`, JSON registry under `HF_LEROBOT_CALIBRATION`.
- Generalizable: works for any UVC camera that exposes a V4L2 capture node, regardless of whether the manufacturer assigned unique serials.

## Out of scope (deferred, not forgotten)

- **Sensor-noise fingerprinting** — academically rigorous (every sensor has unique hot pixels) but heavy implementation; not needed when interactive disambiguation works.
- **Reflashing the Sonix/Microdia iSerial EEPROM** — would give us real unique serials but requires chip-specific tools, is mostly Windows-only, and can soft-brick a camera. Not worth the risk when the picker covers the same ground.
- **RealSense / librealsense integration** — already has unique serials, so the `id` pattern applies cleanly but lives in `RealSenseCameraConfig`, not `OpenCVCameraConfig`. Trivial to add once the OpenCV path lands.
- **Auto-generating udev symlinks** like `/dev/lerobot/cameras/right_overhead` — pure ergonomics for shell scripting; registry alone is sufficient inside LeRobot.
- **Multi-machine portability** — registry is host-local. Moving cameras to a different machine means re-registering on that machine. Acceptable; same constraint as the SO-101 registry.
