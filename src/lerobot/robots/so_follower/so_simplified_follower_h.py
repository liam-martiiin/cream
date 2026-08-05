#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import logging
import time
from functools import cached_property
import serial
from collections import deque
from enum import Enum
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.device_registry import resolve_or_verify_port
from ..utils import ensure_safe_goal_position   # ← correct relative import

from ..robot import Robot
from .config_so_follower import SOSimplifiedFollowerHConfig

logger = logging.getLogger(__name__)

ARDUINO_PORT = "/dev/ttyACM4"
BAUD_RATE = 115200


class DrawingState(Enum):
    MARKER_UP_INIT = 0     # Starting position, marker pointing up
    MARKER_DOWN_AIR = 1    # Marker flipped down, reading decreased due to gravity
    TOUCHING_SURFACE = 2   # Contact made, reading spiked back up


class SOSimplifiedFollowerH(Robot):
    """
    Generic SO follower base implementing common functionality for SO-100/101/10X.
    Designed to be subclassed with a per-hardware-model `config_class` and `name`.
    """

    config_class = SOSimplifiedFollowerHConfig
    name = "so_simplified_follower_h"

    def __init__(self, config: SOSimplifiedFollowerHConfig):
        super().__init__(config)
        self.current_pressure = 0.0

        # --- Touching State Variables ---
        self._pressure_window = deque(maxlen=5)
        self._arm_state = DrawingState.MARKER_UP_INIT
        self._initial_up_baseline = None
        # --------------------------------

        self.config = config
        # FIX: resolve port into a local variable to avoid modifying a possibly frozen config
        port = resolve_or_verify_port(
            self.id, self.config.port, register_command_hint="--type so_simplified_follower_h"
        )

        # FIX: RANGE_M100_100 → RANGE_MINUS100_100
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_MINUS100_100
        self.bus = FeetechMotorsBus(
            port=port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                # "wrist_roll": Motor(5, "sts3215", norm_mode_body),   # removed for simplification
                # "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        # Serial connection to Arduino (pressure sensor)
        self.ser = None
        try:
            self.ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
            time.sleep(2)  # allow Arduino to reset
            self.ser.reset_input_buffer()
        except Exception as e:
            logger.warning(f"Could not connect to Arduino at {ARDUINO_PORT}: {e}")

    @property
    def features(self) -> dict:
        feats = super().features
        # Override observation.state to include 4 motors + pressure + is_touching
        feats["observation.state"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "pressure",
                "is_touching",
            ],
        }
        return feats

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file "
                "or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        # FIX: reset the pressure state machine at the beginning of every session
        self.reset_touch_baseline()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # Record ranges for all motors (no special full-turn motor in this simplified version)
        print("Move all joints sequentially through their entire ranges of motion.")
        print("Recording positions. Press ENTER to stop...")
        all_motors = list(self.bus.motors.keys())
        range_mins, range_maxes = self.bus.record_ranges_of_motion(all_motors)

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                # No gripper-specific settings here

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def is_touching(self, latest_pressure: float) -> bool:
        self._pressure_window.append(latest_pressure)

        if len(self._pressure_window) == 0:
            return False

        smoothed_pressure = sum(self._pressure_window) / len(self._pressure_window)

        # 1. Initialize the starting upward baseline (expected to be ~0)
        if self._initial_up_baseline is None:
            self._initial_up_baseline = smoothed_pressure
            return False

        # --- Thresholds (tune these for your setup) ---
        DOWN_THRESHOLD = self._initial_up_baseline - 50.0
        CONTACT_THRESHOLD = self._initial_up_baseline + 20.0

        # --- STATE MACHINE ---
        if self._arm_state == DrawingState.MARKER_UP_INIT:
            if smoothed_pressure < DOWN_THRESHOLD:
                self._arm_state = DrawingState.MARKER_DOWN_AIR

        elif self._arm_state == DrawingState.MARKER_DOWN_AIR:
            if smoothed_pressure > CONTACT_THRESHOLD:
                self._arm_state = DrawingState.TOUCHING_SURFACE

        # TOUCHING_SURFACE is terminal until reset

        return self._arm_state == DrawingState.TOUCHING_SURFACE

    def reset_touch_baseline(self):
        """Call at the start of a new episode."""
        self._initial_up_baseline = None
        self._pressure_window.clear()
        self._arm_state = DrawingState.MARKER_UP_INIT

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read pressure from Arduino if available
        if self.ser is not None:
            try:
                self.ser.reset_input_buffer()
                self.ser.write(b'P')
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    # FIX: guard against non-numeric strings
                    self.current_pressure = float(line)
            except Exception as e:
                logger.warning(f"Error reading pressure: {e}")
                # keep previous value

        # Fallback if no pressure reading yet
        if not hasattr(self, 'current_pressure'):
            self.current_pressure = 0.0

        # Evaluate touching state
        is_touching_flag = 1.0 if self.is_touching(self.current_pressure) else 0.0

        # Read arm positions
        start = time.perf_counter()
        raw_motor_dict = self.bus.sync_read("Present_Position")

        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
        motor_positions = [raw_motor_dict.get(name, 0.0) for name in motor_names]

        # Build state vector: 4 motor positions + pressure + touch flag
        state_list = motor_positions + [self.current_pressure, is_touching_flag]
        state_array = np.array(state_list, dtype=np.float32)

        obs_dict = {f"{name}.pos": raw_motor_dict.get(name, 0.0) for name in motor_names}
        obs_dict["observation.state"] = state_array

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            # FIX: import now resolved correctly
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def _ticks_to_radians(self, motor_name: str, raw_ticks: int) -> float:
        """Convert raw ticks (0-4095) to radians, centered around 0."""
        import math
        centered = raw_ticks - 2048  # 4096/2
        return (centered / 4096.0) * (2.0 * math.pi)

    def _radians_to_ticks(self, motor_name: str, radians: float) -> int:
        """Convert radians to raw ticks (0-4095)."""
        import math
        centered = int(round(radians * (4096.0 / (2.0 * math.pi))))
        # clamp to avoid overflow
        centered = max(-2048, min(2047, centered))
        return centered + 2048

    @check_if_not_connected
    def get_observation_radians(self) -> dict[str, float]:
        rad_dict = {}
        for motor in self.bus.motors:
            raw_ticks = self.bus.read("Present_Position", motor)
            rad_dict[f"{motor}.pos"] = self._ticks_to_radians(motor, raw_ticks)
        return rad_dict

    @check_if_not_connected
    def send_action_radians(self, action_radians: dict[str, float]) -> None:
        for key, rad_val in action_radians.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                raw_ticks = self._radians_to_ticks(motor_name, rad_val)
                self.bus.write("Goal_Position", motor_name, raw_ticks)

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        logger.info(f"{self} disconnected.")


SOSimplifiedFollowerH = SOSimplifiedFollowerH