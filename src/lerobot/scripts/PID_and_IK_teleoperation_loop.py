import time
import serial
import threading
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# IKPY Import
import ikpy.chain

# Update these imports to match your project's directory structure
from lerobot.teleoperators.so_leader.config_so_leader import SOSimplifiedLeaderPIDConfig
from lerobot.robots.so_follower.config_so_follower import SOSimplifiedFollowerPIDConfig
from lerobot.teleoperators.so_leader.so_simplified_leader_PID import SOSimplifiedLeaderPID
from lerobot.robots.so_follower.so_simplified_follower_PID import SOSimplifiedFollowerPID

stop_event = threading.Event()  

# --- Hardware Configuration ---
ARDUINO_PORT = "/dev/ttyACM0" 
BAUD_RATE = 115200

Kp = 0.000004
Ki = 0.00
Kd = 0.00

# --- Global State Variables ---
current_pressure = 0.0
target_pressure = 0.0
pressure_locked = False
locked_z_pos = 0.0

DEADBAND = 5.0 # MAXIMUM PRESSURE ERROR ALLOWED BEFORE PID OUTPUTS, IN GRAMS
MAX_Z_STEP = 0.0005 # MAXIMUM CHANGE IN Z POSITION PER LOOP ITERATION, IN METERS
ALPHA = 0.15 # LOWER TO MAKE LOW PASS FILTERING MORE AGRESSIVE, HIGHER TO MAKE IT MORE RESPONSIVE

# --- PID Tracking Variables --- 
integral = 0.0
prev_pressure = 0.0
prev_time = time.perf_counter()

# --- GUI State Variables ---
time_history = []
pressure_history = []
gui_start_time = time.perf_counter()

def read_arduino():
    """Background thread to poll the latest load cell pressure via serial."""
    global current_pressure
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2) #SHORTEN AS MUCH AS WE CAN TO SPEED UP STARTUP
        ser.reset_input_buffer()
        
        while True:
            ser.write(b'P')
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    raw_pressure = float(line)
                    current_pressure = raw_pressure * ALPHA + (1-ALPHA) * current_pressure
                except ValueError:
                    continue
            time.sleep(0.01) 
            
    except Exception as e:
        print(f"Warning: Could not connect to Arduino at {ARDUINO_PORT}: {e}")

def listen_for_input():
    """Background thread to listen for 'p' + ENTER to toggle the PID lock."""
    global pressure_locked, target_pressure, current_pressure, integral, prev_pressure, prev_time
    while True:
        user_input = sys.stdin.readline().strip()
        if user_input.lower() == 'p':
            pressure_locked = not pressure_locked
            if pressure_locked:
                target_pressure = current_pressure
                integral = 0.0
                prev_pressure = current_pressure
                prev_time = time.perf_counter()
                print(f"\n\n[LOCKED] Target Pressure Set To: {target_pressure}")
            else:
                print("\n\n[UNLOCKED] Control returned to leader arm.")

def teleoperation_worker():
    global locked_z_pos, integral, prev_pressure, prev_time

    # --- Load URDF for IKPY ---
    try:
        arm_chain = ikpy.chain.Chain.from_urdf_file("src/lerobot/URDF/pressure_sensing_so101.urdf")
        
        # --- NEW: JAX Pre-compilation ---
        # Run a dummy IK calculation so the 1-5 second JIT compilation 
        # happens here instead of freezing the live control loop later.
        print("\nCompiling JAX backend (this may take a few seconds)...")
        dummy_joints = [0.0] * len(arm_chain.links)
        arm_chain.inverse_kinematics(
            target_position=[0.1, 0.1, 0.1], 
            initial_position=dummy_joints, 
            backend="jax"
        )
        print("JAX compilation complete!\n")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load URDF file: {e}")
        return

    threading.Thread(target=read_arduino, daemon=True).start()
    threading.Thread(target=listen_for_input, daemon=True).start()

    print("Connecting hardware...")
    
    lead_config = SOSimplifiedLeaderPIDConfig()
    lead_config.id = "right_leader"
    #lead_config.port = "/dev/ttyACM1"  # Update this to the correct port for your leader arm
    lead_config.cameras = {}
    leader = SOSimplifiedLeaderPID(lead_config)
    
    follow_config = SOSimplifiedFollowerPIDConfig()
    follow_config.id = "right_follower"
    #follow_config.port = "/dev/ttyACM2"  # Update this to the correct port for your follower arm
    follow_config.cameras = {}
    follower = SOSimplifiedFollowerPID(follow_config)
    
    follower.connect()
    leader.connect()
    
    last_print_time = time.perf_counter()
    
    # Track the arm's state for the JAX "Warm Start"
    last_follower_angles_rad = [0.0] * len(arm_chain.links)
    
    try:
        while not stop_event.is_set():
        
            
            time_history.append(time.perf_counter() - gui_start_time)
            pressure_history.append(current_pressure)

            action = leader.get_action()

            leader_angles_rad = [
                0.0, 
                math.radians(action.get('shoulder_pan.pos', 0.0)),
                math.radians(action.get('shoulder_lift.pos', 0.0)),
                math.radians(action.get('elbow_flex.pos', 0.0)),
                math.radians(action.get('wrist_flex.pos', 0.0)),
                0.0
            ]

            # Use JAX for Forward Kinematics as well[cite: 1]
            leader_frame = arm_chain.forward_kinematics(leader_angles_rad, backend="jax")
            leader_x = leader_frame[0, 3]
            leader_y = leader_frame[1, 3]
            leader_z = leader_frame[2, 3]
        
            if pressure_locked:
                current_time = time.perf_counter()
                dt = current_time - prev_time
                
                if dt > 0:
                    error = target_pressure - current_pressure
                    
                    derivative = - (current_pressure - prev_pressure) / dt #calculated before deadband
                    
                    if(abs(error)<DEADBAND):
                        error = 0
                    
                    proportional = Kp * error
                    integral += (error * dt)
                    
                    
                    output = proportional + (Ki * integral) + (Kd * derivative)
                    
                    output = max(-MAX_Z_STEP, min(MAX_Z_STEP, output))
                    locked_z_pos -= output 
                    
                    prev_pressure = current_pressure
                    prev_time = current_time
            else:
                locked_z_pos = leader_z
            
            target_position = [leader_x, leader_y, locked_z_pos]
            
            # --- NEW: Prioritize ALL Leader Joint Angles ---
            # Copy the entire leader arm's state to use as the starting guess.
            # IKPY's optimizer will naturally find the solution closest to this pose.
            ideal_initial_guess = leader_angles_rad.copy()
            
            # --- SAFETY CLAMP: Enforce URDF Joint Limits ---
            # We still need to protect against SciPy bounds errors in case the 
            # leader arm briefly physically exceeds the follower's URDF limits.
            for i, link in enumerate(arm_chain.links):
                if link.bounds is not None:
                    min_bound, max_bound = link.bounds
                    ideal_initial_guess[i] = max(
                        min_bound + 1e-6, 
                        min(max_bound - 1e-6, ideal_initial_guess[i])
                    )
            
            # --- JAX Inverse Kinematics with Warm Start ---
            follower_angles_rad = arm_chain.inverse_kinematics(
                target_position=target_position,
                initial_position=ideal_initial_guess,
                backend="jax"
            )
            
            # (Optional) You can still save this if you need it for other logic,
            # but the next iteration will now always base its guess on the leader.
            last_follower_angles_rad = follower_angles_rad

            action['shoulder_pan.pos'] = math.degrees(follower_angles_rad[1])
            action['shoulder_lift.pos'] = math.degrees(follower_angles_rad[2])
            action['elbow_flex.pos'] = math.degrees(follower_angles_rad[3])
            action['wrist_flex.pos'] = math.degrees(follower_angles_rad[4])

            follower.send_action(action)
            
            current_time_loop = time.perf_counter()
            if current_time_loop - last_print_time >= 0.1:
                status_text = f"LOCKED @ {target_pressure:.1f}" if pressure_locked else "UNLOCKED"
                sys.stdout.write(f"\rLive Pressure: {current_pressure:>6.1f} | Status: {status_text:<20}")
                sys.stdout.flush()
                last_print_time = current_time_loop
            
            time.sleep(0.01)
            
    finally:
        print("\n\nDisconnecting...")
        follower.disconnect()
        leader.disconnect()

def update_plot(frame, line, ax):
    if len(time_history) > 0:
        line.set_data(time_history, pressure_history)
        current_time = time_history[-1]
        ax.set_xlim(0, max(10, current_time + 1))
    return line,

def main():
    teleop_thread = threading.Thread(target=teleoperation_worker)
    teleop_thread.start()

    fig, ax = plt.subplots()
    ax.set_ylim(-500, 500)
    ax.set_title("Live Load Cell Pressure")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure")
    ax.grid(True)
    
    line, = ax.plot([], [], lw=2, color='blue')
    
    ani = FuncAnimation(fig, update_plot, fargs=(line, ax), interval=100, blit=False, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nClosing GUI and signaling arm to shut down...")
        stop_event.set()       
        teleop_thread.join()   

if __name__ == "__main__":
    main()