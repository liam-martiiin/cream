"""
record_pressure_ik.py

Drop-in replacement for `lerobot-record` that adds pressure-locked,
JAX-accelerated IK teleop (ported from PID_and_IK_teleoperation_loop.py)
directly into the dataset-recording loop.

USAGE (same flags as lerobot-record, plus your robot/teleop types):

    python record_pressure_ik.py \
        --robot.type=so_follower_pid \
        --robot.id=right_follower \
        --teleop.type=so_leader_pid \
        --teleop.id=right_leader \
        --dataset.repo_id=<user>/<dataset_name> \
        --dataset.single_task="Pick the cube" \
        --dataset.num_episodes=10 \
        --dataset.fps=30 \
        --display_data=true

Update ROBOT_TYPE / TELEOP_TYPE placeholders below to whatever strings
your SOSimplifiedFollowerPID / SOSimplifiedLeaderPID are registered
under (check config_so_follower.py / config_so_leader.py for the
`type` field used in @RobotConfig.register_subclass(...)).

WHY MONKEY-PATCHING:
lerobot_record.record() does a lot of work we don't want to reimplement
(dataset creation/resume, episode loop, re-record/reset handling, hub
push, keyboard listener wiring, image-writer lifecycle). Its internal
calls to `record_loop(...)` resolve that name from the module's own
global namespace at call time, so replacing
`lerobot.scripts.lerobot_record.record_loop` before calling `.record()`
is sufficient to swap in our version everywhere it's used (recording
AND reset phases) with zero other changes.
"""

import logging
import math
import sys
import threading
import time

import serial

import ikpy.chain

import lerobot.scripts.lerobot_record as lr_record
from lerobot.datasets import safe_stop_image_writer
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_rerun_data

# --- Register your custom robot/teleop config+class if they aren't
# already registered elsewhere in your package __init__ files. If
# `so_follower_pid` / `so_leader_pid` (or whatever you name them) are
# already discoverable via lerobot.robots / lerobot.teleoperators
# __init__.py, you can delete this import block.
from lerobot.teleoperators.so_leader.config_so_leader import SOSimplifiedLeaderPIDConfig
from lerobot.robots.so_follower.config_so_follower import SOSimplifiedFollowerPIDConfig
from lerobot.teleoperators.so_leader.so_simplified_leader_PID import SOSimplifiedLeaderPID
from lerobot.robots.so_follower.so_simplified_follower_PID import SOSimplifiedFollowerPID

# =====================================================================
# --- Hardware / control constants (unchanged from your teleop script)
# =====================================================================
ARDUINO_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
URDF_PATH = "src/lerobot/URDF/pressure_sensing_so101.urdf"

Kp = 0.000002
Ki = 0.00
Kd = 0.000000000

MIN_Z_POS = -0.3
MAX_Z_POS = 0.3

DEADBAND = 5.0       # grams of pressure error before PID responds
MAX_Z_STEP = 0.0005  # meters per loop iteration
ALPHA = 0.15         # low-pass filter coefficient for pressure readings

# =====================================================================
# --- Shared state between the Arduino thread, the 'p'-key toggle
#     thread, and the record loop
# =====================================================================
current_pressure = 0.0
target_pressure = 0.0
pressure_locked = False
locked_z_pos = 0.0

integral = 0.0
prev_pressure = 0.0
prev_time = time.perf_counter()

arm_chain = None  # populated by init_ik_chain()


def init_ik_chain():
    """Load the URDF and JIT-warm the JAX IK/FK backend once, before
    recording starts, so the first frame of the first episode isn't
    stuck paying the 1-5s compile cost."""
    global arm_chain
    print("Loading URDF and compiling JAX backend (this may take a few seconds)...")
    arm_chain = ikpy.chain.Chain.from_urdf_file(URDF_PATH)
    dummy_joints = [0.0] * len(arm_chain.links)
    arm_chain.forward_kinematics(dummy_joints, backend="jax")
    arm_chain.inverse_kinematics(
        target_position=[0.1, 0.1, 0.1],
        initial_position=dummy_joints,
        backend="jax",
    )
    print("JAX compilation complete.")


def read_arduino():
    """Background thread: poll the load cell over serial, low-pass filter it."""
    global current_pressure
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
        while True:
            ser.write(b"P")
            line = ser.readline().decode("utf-8").strip()
            if line:
                try:
                    raw_pressure = float(line)
                    current_pressure = raw_pressure * ALPHA + (1 - ALPHA) * current_pressure
                except ValueError:
                    continue
            time.sleep(0.01)
    except Exception as e:
        print(f"Warning: Could not connect to Arduino at {ARDUINO_PORT}: {e}")


def listen_for_input():
    """Background thread: type 'p' + ENTER in the terminal to toggle the
    pressure lock. This runs alongside (not instead of) lerobot's own
    pynput arrow-key listener for episode control, since it reads from
    stdin rather than OS-level key hooks."""
    global pressure_locked, target_pressure, current_pressure, integral, prev_pressure, prev_time
    while True:
        user_input = sys.stdin.readline().strip()
        if user_input.lower() == "p":
            pressure_locked = not pressure_locked
            if pressure_locked:
                target_pressure = current_pressure
                integral = 0.0
                prev_pressure = current_pressure
                prev_time = time.perf_counter()
                print(f"\n[LOCKED] Target Pressure Set To: {target_pressure}")
            else:
                print("\n[UNLOCKED] Control returned to leader arm.")


def apply_pressure_lock_and_ik(action: dict) -> dict:
    """Core of the original teleoperation loop's math, factored into a
    pure-ish function: reads the leader's raw action dict, runs FK to
    find the leader's task-space pose, applies the pressure-lock PID
    to the Z axis if locked, solves IK for the follower, and overwrites
    the joint angle keys in `action` in place with the follower's
    solution (so the dataset records what the follower actually did).
    """
    global locked_z_pos, integral, prev_pressure, prev_time

    leader_angles_rad = [
        0.0,
        math.radians(action.get("shoulder_pan.pos", 0.0)),
        math.radians(action.get("shoulder_lift.pos", 0.0)),
        math.radians(action.get("elbow_flex.pos", 0.0)),
        math.radians(action.get("wrist_flex.pos", 0.0)),
        0.0,
    ]

    leader_frame = arm_chain.forward_kinematics(leader_angles_rad, backend="jax")
    leader_x = leader_frame[0, 3]
    leader_y = leader_frame[1, 3]
    leader_z = leader_frame[2, 3]

    if pressure_locked:
        current_time = time.perf_counter()
        dt = current_time - prev_time
        if dt > 0:
            error = target_pressure - current_pressure
            derivative = -(current_pressure - prev_pressure) / dt  # pre-deadband

            if abs(error) < DEADBAND:
                error = 0

            proportional = Kp * error
            integral += error * dt
            output = proportional + (Ki * integral) + (Kd * derivative)
            output = max(-MAX_Z_STEP, min(MAX_Z_STEP, output))

            locked_z_pos -= output
            locked_z_pos = max(MIN_Z_POS, min(MAX_Z_POS, locked_z_pos))

            prev_pressure = current_pressure
            prev_time = current_time
    else:
        locked_z_pos = leader_z

    target_position = [leader_x, leader_y, locked_z_pos]

    ideal_initial_guess = leader_angles_rad.copy()
    for i, link in enumerate(arm_chain.links):
        if link.bounds is not None:
            min_bound, max_bound = link.bounds
            ideal_initial_guess[i] = max(min_bound + 1e-6, min(max_bound - 1e-6, ideal_initial_guess[i]))

    follower_angles_rad = arm_chain.inverse_kinematics(
        target_position=target_position,
        initial_position=ideal_initial_guess,
        backend="jax",
    )

    action["shoulder_pan.pos"] = math.degrees(follower_angles_rad[1])
    action["shoulder_lift.pos"] = math.degrees(follower_angles_rad[2])
    action["elbow_flex.pos"] = math.degrees(follower_angles_rad[3])
    action["wrist_flex.pos"] = math.degrees(follower_angles_rad[4])

    return action


# =====================================================================
# --- Patched record_loop: identical to lerobot_record.record_loop
#     except for the block marked "PRESSURE + IK" below.
# =====================================================================
@safe_stop_image_writer
def record_loop_pressure_ik(
    robot,
    events: dict,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    dataset=None,
    teleop=None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(t, (so_leader.SO100Leader | so_leader.SO101Leader | koch_leader.KochLeader | omx_leader.OmxLeader))
            ),
            None,
        )
        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm "
                "teleoperator. Currently only supported for LeKiwi robot. Pressure-lock IK is not "
                "implemented for multi-teleop setups."
            )

    control_interval = 1 / fps
    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        if isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            # ---------------- PRESSURE + IK ----------------
            # Replace the leader's raw joint targets with the
            # pressure-locked, IK-solved follower targets before they
            # flow through the normal processor pipeline. This mirrors
            # your original loop's in-place overwrite of `action`.
            if arm_chain is not None:
                act = apply_pressure_lock_and_ik(act)
            # -------------------------------------------------

            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        elif isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        _sent_action = robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        # live status line (same as your original script)
        status_text = f"LOCKED @ {target_pressure:.1f}" if pressure_locked else "UNLOCKED"
        sys.stdout.write(f"\rLive Pressure: {current_pressure:>6.1f} | Status: {status_text:<20}")
        sys.stdout.flush()

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). "
                "Dataset frames might be dropped and robot control might be unstable."
            )
        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


def main():
    lr_record.register_third_party_plugins()

    # Patch the module-level name that lerobot_record.record() calls.
    lr_record.record_loop = record_loop_pressure_ik

    init_ik_chain()
    threading.Thread(target=read_arduino, daemon=True).start()
    threading.Thread(target=listen_for_input, daemon=True).start()

    lr_record.record()


if __name__ == "__main__":
    main()
